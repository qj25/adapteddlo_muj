"""Shared MuJoCo wire plugin XML config for rope models."""

from typing import Dict, Optional, Tuple

COSSERAT_WIRE_PLUGIN_CONFIGS = {"fullDyn": "true"}

NATIVE_ROPE_XML = "nativerope1dkin.xml"
NATIVE_OVERALL_XML = "overall_native.xml"
DLO_ROPE_XML = "dlorope1dkin.xml"
DLO_OVERALL_XML = "overall.xml"


def uses_native_xml_paths(extra_plugin_configs: Optional[Dict[str, str]]) -> bool:
    """Wire-plugin models with fullDyn use the same saved XML paths as native cable."""
    return (
        extra_plugin_configs is not None
        and extra_plugin_configs.get("fullDyn", "").lower() == "true"
    )


def rope_xml_paths(
    plugin_name: str,
    extra_plugin_configs: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    if plugin_name == "cable" or uses_native_xml_paths(extra_plugin_configs):
        return NATIVE_ROPE_XML, NATIVE_OVERALL_XML
    return DLO_ROPE_XML, DLO_OVERALL_XML
