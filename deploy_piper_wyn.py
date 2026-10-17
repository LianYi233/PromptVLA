#!/usr/bin/env python3
"""Compatibility entry point for the Piper deployment command used on the robot PC.

Delegate to the maintained entry point, including its checkout-source selection,
environment checks, and checkpoint validation. Keep deployment logic in one file.
"""

from deploy_piper_vlaa import main

if __name__ == "__main__":
    main()
