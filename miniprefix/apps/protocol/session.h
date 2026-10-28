#pragma once
#include "miniprefix/engine.h"
#include <istream>
#include <ostream>
namespace miniprefix::protocol {
// Line loop only. Startup/load failures remain the executable's responsibility.
int serve(Engine&,std::istream&,std::ostream&);
}
