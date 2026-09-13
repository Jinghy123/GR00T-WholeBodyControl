import getpass
import os

# Per-user trace path. /tmp/cdds.LOG is created by whichever account first runs
# the SDK; any other user then fails "cannot open for writing" and Cyclone DDS
# refuses to create the domain at all. Override with CDDS_TRACE_FILE if needed.
ChannelConfigTraceFile = os.environ.get(
    "CDDS_TRACE_FILE", f"/tmp/cdds.{getpass.getuser()}.LOG"
)

ChannelConfigHasInterface = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
                </Interfaces>
            </General>
            <Tracing>
                <Verbosity>config</Verbosity>
            <OutputFile>$__TRACE_FILE__$</OutputFile>
        </Tracing>
        </Domain>
    </CycloneDDS>'''

ChannelConfigAutoDetermine = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface autodetermine=\"true\" priority=\"default\" multicast=\"default\" />
                </Interfaces>
            </General>
        </Domain>
    </CycloneDDS>'''
