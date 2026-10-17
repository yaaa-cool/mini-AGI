#!/bin/bash
# config.sh [small|reference] : (re)apply the rig's run config after a sync,
# or change it. With no argument it re-applies the recorded active config.
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
want=${1:-$(active_config)} || exit 2
select_config "$want"
