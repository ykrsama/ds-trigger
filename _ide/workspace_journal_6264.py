# 2025-11-30T18:11:29.278510900
import vitis

client = vitis.create_client()
client.set_workspace(path="ds-trigger")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

vitis.dispose()

