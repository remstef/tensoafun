#!/bin/sh
exec scala -J-Xmx2g "$1" "${@:2}"
!#