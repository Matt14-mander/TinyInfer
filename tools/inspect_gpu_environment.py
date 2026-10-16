#!/usr/bin/env python3
"""Read-only GPU/SDK inventory. Does not install packages or alter device settings."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


def probe(command, timeout=20):
    if not shutil.which(command[0]):
        return {'status': 'not_found', 'command': command}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {'status': 'ok' if result.returncode == 0 else 'failed',
                'command': command, 'returncode': result.returncode,
                'stdout': result.stdout[:24000], 'stderr': result.stderr[:4000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'unavailable', 'command': command, 'reason': str(error)}


def metal_devices():
    # Enumerate devices and compile a trivial library; no GPU kernel is executed.
    source = '''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main() { @autoreleasepool {
    NSMutableArray *items = [NSMutableArray array];
    for (id<MTLDevice> device in MTLCopyAllDevices()) {
        NSError *error = nil;
        NSString *source = @"#include <metal_stdlib>\\nusing namespace metal;\\n"
            "kernel void tinyinfer_probe(device float *out [[buffer(0)]], "
            "uint i [[thread_position_in_grid]]) { out[i] = 0.0f; }";
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        NSString *message = error ? error.localizedDescription : @"";
        [items addObject:@{@"name":device.name, @"low_power":@(device.lowPower),
                          @"queue_available":@([device newCommandQueue] != nil),
                          @"runtime_shader_compilation":@(library != nil),
                          @"shader_compilation_error":message}];
    }
    NSData *data = [NSJSONSerialization dataWithJSONObject:items options:0 error:nil];
    fwrite(data.bytes, 1, data.length, stdout);
    return 0;
} }
'''
    with tempfile.TemporaryDirectory(prefix='tinyinfer-metal-probe-') as temp:
        path = Path(temp)
        (path / 'probe.m').write_text(source)
        compilation = probe(['xcrun', 'clang', '-fobjc-arc', str(path / 'probe.m'),
                             '-framework', 'Foundation', '-framework', 'Metal',
                             '-o', str(path / 'probe')])
        if compilation['status'] != 'ok':
            return {'status': 'unverified', 'compilation': compilation}
        execution = probe([str(path / 'probe')])
        if execution['status'] != 'ok':
            return {'status': 'unverified', 'execution': execution}
        try:
            devices = json.loads(execution['stdout'])
            return {'status': 'available' if devices else 'no_devices', 'devices': devices}
        except json.JSONDecodeError:
            return {'status': 'unverified', 'execution': execution}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'collected_utc': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'machine': platform.machine(),
              'cuda_driver_inventory': probe(['nvidia-smi']),
              'cuda_toolkit': probe(['nvcc', '--version']),
              'cuda_gpu_details': probe(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
              'cmake': probe(['cmake', '--version']),
              'compiler': probe(['clang', '--version']),
              'host_cxx_compiler': probe(['c++', '--version']),
              'scope': 'Hardware/toolchain inventory only; no TinyInfer GPU kernel acceptance.'}
    if platform.system() == 'Darwin':
        report.update(macos=probe(['sw_vers']), developer_directory=probe(['xcode-select', '-p']),
                      macos_sdk=probe(['xcrun', '--sdk', 'macosx', '--show-sdk-path']),
                      metal_offline_compiler=probe(['xcrun', '--find', 'metal']),
                      metal_runtime=metal_devices())
    elif platform.system() == 'Windows':
        report['windows_msvc_path'] = probe(['cmd', '/c', 'where', 'cl'])
        report['windows_cpu'] = probe(['powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json'])
        report['windows_gpu'] = probe(['powershell', '-NoProfile', '-Command',
            'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json'])
    output = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    print(output, end='')


if __name__ == '__main__':
    main()
