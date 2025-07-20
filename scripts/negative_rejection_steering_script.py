import gradio as gr
import logging
import sys
from functools import partial
from modules import scripts, script_callbacks
from typing import Any

from NRS.nodes_NRS import NRS

class NRSScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.skew = 4.0
        self.stretch = 2.0
        self.squash = 0.0

    sorting_priority = 5

    def title(self):
        return "Negative Rejection Steering"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enable NRS", value=self.enabled)
            gr.HTML("<p><i>Adjust the settings for Negative Rejection Steering.</i></p>")
            skew = gr.Slider(label="NRS Skew Scale", info="Adjusts the amount guidance is steered.", minimum=-30.0, maximum=30.0, step=0.01, value=self.skew)
            stretch = gr.Slider(label="NRS Stretch Scale", info="Adjusts the amount guidance is amplified.", minimum=-30.0, maximum=30.0, step=0.01, value=self.stretch)
            squash = gr.Slider(label="NRS Squash Multiplier", info="Adjusts the amount final guidance is normalized.", minimum=0.0, maximum=1.0, step=0.01, value=self.squash)

        enabled.change(
            lambda x: self.update_enabled(x),
            inputs=[enabled]
        )

        return (enabled, skew, stretch, squash)
    

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 4:
            self.enabled, self.skew, self.stretch, self.squash = args[:4]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        xyz = getattr(p, "_nrs_xyz", {})
        if "enabled" in xyz:
            self.enabled = xyz["enabled"] == "True"
        if "skew" in xyz:
            self.skew = xyz["skew"]
        if "stretch" in xyz:
            self.stretch = xyz["stretch"]
        if "squash" in xyz:
            self.squash = xyz["squash"]

        # Always start with a fresh clone of the original unet
        unet = p.sd_model.forge_objects.unet.clone()

        if not self.enabled:
            # Reset the unet to its original state
            p.sd_model.forge_objects.unet = unet
            return

        unet = NRS().patch(unet, self.skew, self.stretch, self.squash)[0]

        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params.update({
            "NRS_enabled": True,
            "NRS_skew": self.skew,
            "NRS_stretch": self.stretch,
            "NRS_squash": self.squash,
        })

        logging.debug(f"NRS: Enabled: {self.enabled}, Squash: {self.skew}, Stretch: {self.stretch}, Squash: {self.squash}")

        return

def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_nrs_xyz"):
        p._nrs_xyz = {}
    p._nrs_xyz[field] = x

def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption(
            "(NRS) Enabled",
            str,
            partial(set_value, field="enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(NRS) Skew",
            float,
            partial(set_value, field="skew"),
        ),
        xyz_grid.AxisOption(
            "(NRS) Stretch",
            float,
            partial(set_value, field="stretch"),
        ),
        xyz_grid.AxisOption(
            "(NRS) Squash",
            float,
            partial(set_value, field="squash"),
        ),
    ]

    if not any(x.label.startswith("(NRS)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] NRS Script: xyz_grid error:\n{error}",
            file=sys.stderr,
        )

script_callbacks.on_before_ui(on_before_ui)
