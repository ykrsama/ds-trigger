# 2025-11-30T17:40:24.966296100
import vitis

client = vitis.create_client()
client.set_workspace(path="ds-trigger")

comp = client.create_hls_component(name = "hls_component",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

