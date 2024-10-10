# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: embed
#     language: python
#     name: python3
# ---

import MAP_VAE.utils.seq_tools as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from MAP_VAE.utils.datasets import MSA_Dataset
from MAP_VAE.models.seqVAE import SeqVAE
import yaml
import torch
import MAP_VAE.utils.statistics as stats
import MAP_VAE.utils.visualisation as vs
from ipydatagrid import DataGrid, TextRenderer, BarRenderer, Expr, ImageRenderer
from ipywidgets import (
    FloatSlider,
    Dropdown,
    ColorPicker,
    HBox,
    VBox,
    widgets,
    Output,
    IntSlider,
)
from ipydatagrid import DataGrid, TextRenderer, Expr
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
import logomaker
from IPython.display import display, clear_output
import numpy as np


# +


def perform_reconstruction(
    sequences: pd.DataFrame, model: SeqVAE, device
) -> pd.DataFrame:
    """
    Reconstructs sequences using a SeqVAE model.

    Args:
        sequences (pd.DataFrame): A DataFrame containing the sequences to be reconstructed.
        model (SeqVAE): The SeqVAE model used for reconstruction.
        device: The device (e.g., "cpu", "cuda") on which the model will be run.

    Returns:
        pd.DataFrame: A DataFrame containing the reconstructed sequences and their corresponding IDs.
    """
    encodings = sequences["sequence"].apply(st.seq_to_one_hot)
    dataset = MSA_Dataset(
        encodings, np.arange(sequences.shape[0]), sequences["id"], device
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    recons = []
    ids = []
    for x, _, id in loader:
        x = x.to(device)
        indices = model.reconstruct(x)

        # how to have two loops in list comprehension
        sequences = [
            "".join([st.GAPPY_PROTEIN_ALPHABET[i] for i in seq]) for seq in indices
        ]
        recons.extend(sequences)
        ids.extend(id)

    df = pd.DataFrame({"id": ids, "sequence": recons})
    return df


def prepare_model(sequences: pd.DataFrame, state_dict: str, settings: dict) -> SeqVAE:

    seq_len = len(sequences["sequence"].values[0])
    input_dims = seq_len * st.GAPPY_ALPHABET_LEN

    device = torch.device("mps")
    model = SeqVAE(
        dim_latent_vars=settings["latent_dims"],
        dim_msa_vars=input_dims,
        num_hidden_units=settings["hidden_dims"],
        settings=settings,
        num_aa_type=21,
    )

    model.load_state_dict(torch.load(state_dict, map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    return model


# Function to update the expression and DataGrid
def update_colour(*args, **kwargs):

    kwargs["conditional_expression"].value = (
        "'{color}' if cell.value {operator} {highlight} else default_value".format(
            operator=kwargs["operator_dropdown"].value,
            highlight=kwargs["highlight"].value,
            color=kwargs["output_colorpicker"].value,
        )
    )


def update_datagrid(sequence_log_p: np.ndarray):

    to_plot = sequence_log_p[0, :, :]

    to_plot = to_plot.T

    data = {"data": [], "schema": {}}

    data["data"] = to_plot.tolist()

    data["schema"]["fields"] = [
        {"name": c, type: "number"} for c in st.GAPPY_PROTEIN_ALPHABET
    ]

    return pd.DataFrame(
        data["data"], index=st.GAPPY_PROTEIN_ALPHABET
    )  # , new_logo_data


# +


# Define the rendering function
def renderer_function(cell, default_value):
    return "#fc8403" if cell.value > 0.2 else default_value


aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/all_alns/"
# aln_path = "/Users/sebs_mac/uni_OneDrive/honours/data/cassowary/vis/"
extants = st.read_aln_file(aln_path + "tree_seqs/gcn4_tree_0_ancestors_extants.aln")

state_dict = f"/Users/sebs_mac/uni_OneDrive/honours/data/optimised_model_metrics/zero_shot_tasks/model_states/gcn4_ae_r1_model_state.pt"
state_dict = (
    f"/Users/sebs_mac/uni_OneDrive/honours/data/vis_models/gcn4_ae_r1_model_state.pt"
)

with open("../data/dummy_config.yaml", "r") as stream:
    settings = yaml.safe_load(stream)
settings["latent_dims"] = 3

model = prepare_model(extants, state_dict, settings)

device = torch.device("mps")
encodings = extants["sequence"].apply(st.seq_to_one_hot)
dataset = MSA_Dataset(encodings, np.arange(extants.shape[0]), extants["id"], device)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Initial setup
x, _, _ = iter(loader).__next__()  # grab first sequence
log_p = model.get_log_p(x)

to_plot = np.exp(log_p[0, :, :]).T

data = {"data": [], "schema": {}}
data["data"] = to_plot.tolist()
data["schema"]["fields"] = [
    {"name": c, type: "number"} for c in st.GAPPY_PROTEIN_ALPHABET
]

df = pd.DataFrame(data["data"], index=st.GAPPY_PROTEIN_ALPHABET)  # , new_logo_data

ids = []
latent = []

with torch.no_grad():
    for x, _, id in loader:
        x = torch.flatten(x, start_dim=1)
        mu, sigma = model.encoder(x)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        z = z.detach().cpu().numpy()[0]
        latent.append(z)

latent = np.stack(latent)

# -


len(extants["sequence"][0])

# +
from ipywidgets import (
    interact,
    HBox,
    VBox,
    Output,
    HTML,
    Dropdown,
    Button,
    Layout,
    Label,
    Box,
)
from IPython.display import display, clear_output
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import logomaker
from torch import layout  # Assuming you are using logomaker for sequence logos


class demo:

    def __init__(
        self,
        input_df: pd.DataFrame,
        latent_space,
        seq_len,
        data_grid_min,
        data_grid_max,
    ):

        self.data_grid_min = data_grid_min
        self.data_grid_max = data_grid_max

        self.seq_len = seq_len
        self.conditional_expression = Expr(renderer_function)
        self.default_renderer = TextRenderer(
            background_color=self.conditional_expression, format=".3f"
        )
        input_df = input_df.iloc[:, self.data_grid_min : self.data_grid_max]
        self.datagrid = DataGrid(input_df, default_renderer=self.default_renderer)
        self.latent_space = latent_space

        self.operator_dropdown = Dropdown(options=["<", ">"], value="<")
        self.reference_slider_1 = FloatSlider(
            value=0.5, min=-10, max=10, description="Z1"
        )
        self.reference_slider_2 = FloatSlider(
            value=0.5, min=-10, max=10, description="Z2"
        )
        self.reference_slider_3 = FloatSlider(
            value=0.5, min=-10, max=10, description="Z3"
        )
        self.highlight = FloatSlider(value=0.5, min=0, max=1, description="Highlight")
        self.output_colorpicker = ColorPicker(value="#fc8403")

        self.operator_dropdown.observe(
            lambda change: self.update_colour(
                change,
                conditional_expression=self.conditional_expression,
                operator_dropdown=self.operator_dropdown,
                highlight=self.highlight,
                output_colorpicker=self.output_colorpicker,
            ),
            "value",
        )

        self.reference_slider_1.observe(
            lambda change: self.update_probs(change, datagrid=self.datagrid), "value"
        )
        self.reference_slider_2.observe(
            lambda change: self.update_probs(change, datagrid=self.datagrid), "value"
        )
        self.reference_slider_3.observe(
            lambda change: self.update_probs(change, datagrid=self.datagrid), "value"
        )

        self.highlight.observe(
            lambda change: self.update_colour(
                change,
                conditional_expression=self.conditional_expression,
                operator_dropdown=self.operator_dropdown,
                highlight=self.highlight,
                output_colorpicker=self.output_colorpicker,
            ),
            "value",
        )

        self.output_colorpicker.observe(
            lambda change: self.update_colour(
                change,
                conditional_expression=self.conditional_expression,
                operator_dropdown=self.operator_dropdown,
                highlight=self.highlight,
                output_colorpicker=self.output_colorpicker,
            ),
            "value",
        )

        self.button_panel = Output(layout=Layout(width="800px"))
        self.prob_panel = Output(layout=Layout(width="800px"))
        self.graph_panel = Output(layout=Layout(width="800px"))

        self.sequnce_panel = Output(layout=Layout(width="800px"))

        # # data grid setup
        hbox = HBox(
            (
                self.reference_slider_1,
                self.reference_slider_2,
                self.reference_slider_3,
                self.operator_dropdown,
                self.highlight,
                self.output_colorpicker,
            )
        )
        self.datagrid_output = VBox([self.datagrid, hbox])

        button_box = VBox([self.button_panel, self.prob_panel])

        scene = VBox(
            [
                HBox([button_box, Box(layout=Layout(width="400px")), self.graph_panel]),
                self.datagrid_output,
                self.sequnce_panel,
            ]
        )

        display(scene)

        with self.button_panel:
            self.btn = Button(description="Update profile")
            self.btn.on_click(self.generate_logo)

            self.start = widgets.BoundedIntText(
                value=0, min=0, max=self.seq_len, step=1, description="Start:"
            )
            self.end = widgets.BoundedIntText(
                value=10, min=0, max=self.seq_len, step=1, description="End:"
            )

            display(HBox([self.btn, self.start, self.end]))

        with self.graph_panel:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                self.latent_space[:, 0],
                self.latent_space[:, 1],
                self.latent_space[:, 2],
            )
            ax.scatter(
                self.reference_slider_1.value,
                self.reference_slider_2.value,
                self.reference_slider_3.value,
                marker="*",
                s=100,
                c="r",
            )
            ax.set_xlabel("Z1")
            ax.set_ylabel("Z2")
            # ax.set_zlabel("Z3", labelpad=0.5)
            ax.set_zlabel("Z3", rotation=90, labelpad=0.5)

            plt.show()

        with self.sequnce_panel:

            self.seq_input = widgets.Textarea(
                value="",
                placeholder="Enter sequence",
                description="Input:",
                disabled=False,
            )

            self.seq_btn = Button(description="Update")
            self.seq_btn.on_click(self.update_sequence)

            display(HBox([self.seq_input, self.seq_btn]))

    def generate_logo(self, x):

        with self.prob_panel:
            clear_output()
            # print(self.datagrid.data.T)
            # print(self.datagrid.data)
            logo_df = self.datagrid.data.iloc[
                :,
                self.start.value
                - self.data_grid_min : self.end.value
                - self.data_grid_min,
            ].T
            # logo_df = self.datagrid.data.T.iloc[self.start.value:self.end.value, :]
            # logo_df = self.datagrid.data.T.iloc[self.start.value - self.data_grid_min:self.end.value - self.data_grid_max, :]
            logo_df = logo_df.rename_axis("pos")

            logo = logomaker.Logo(
                logo_df,
                color_scheme="skylign_protein",
                font_name="Arial Rounded MT Bold",
                show_spines=False,
            )
            plt.show()

        return

    def update_sequence(self, x):

        cleaned = self.seq_input.value.replace("\n", "")
        df = pd.DataFrame({"id": ["new"], "sequence": [cleaned]})
        encodings = df["sequence"].apply(st.seq_to_one_hot)

        dataset = MSA_Dataset(encodings, np.arange(df.shape[0]), df["id"], device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for x, _, id in loader:
                x = torch.flatten(x, start_dim=1)
                mu, sigma = model.encoder(x)
                eps = torch.randn_like(mu)
                z = mu + sigma * eps
                z = z.detach().cpu().numpy()[0]

        self.reference_slider_1.value = z[0]
        self.reference_slider_2.value = z[1]
        self.reference_slider_3.value = z[2]

        log_p = model.latent_to_log_p(
            torch.Tensor(
                [
                    self.reference_slider_1.value,
                    self.reference_slider_2.value,
                    self.reference_slider_3.value,
                ]
            ).to(device),
            self.seq_len,
            21,
        )
        p = np.exp(log_p)
        df = self.update_datagrid(p)
        df = df.iloc[:, self.data_grid_min : self.data_grid_max]
        self.datagrid.data = df

        with self.graph_panel:
            clear_output(wait=True)
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")
            # fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
            ax.scatter(
                self.latent_space[:, 0],
                self.latent_space[:, 1],
                self.latent_space[:, 2],
            )
            ax.scatter(
                self.reference_slider_1.value,
                self.reference_slider_2.value,
                self.reference_slider_3.value,
                marker="*",
                s=100,
                c="r",
            )

            ax.set_xlabel("Z1")
            ax.set_ylabel("Z2")
            ax.set_zlabel("Z3", rotation=90, labelpad=0.5)

            plt.show()

    # Function to update the expression and DataGrid
    def update_colour(self, *args, **kwargs):

        kwargs["conditional_expression"].value = (
            "'{color}' if cell.value {operator} {highlight} else default_value".format(
                operator=kwargs["operator_dropdown"].value,
                highlight=kwargs["highlight"].value,
                color=kwargs["output_colorpicker"].value,
            )
        )

    def update_datagrid(self, sequence_log_p: np.ndarray):

        to_plot = sequence_log_p[0, :, :]

        to_plot = to_plot.T

        data = {"data": [], "schema": {}}

        data["data"] = to_plot.tolist()

        data["schema"]["fields"] = [
            {"name": c, type: "number"} for c in st.GAPPY_PROTEIN_ALPHABET
        ]

        return pd.DataFrame(
            data["data"], index=st.GAPPY_PROTEIN_ALPHABET
        )  # , new_logo_data

    def update_probs(self, *args, **kwargs):

        log_p = model.latent_to_log_p(
            torch.Tensor(
                [
                    self.reference_slider_1.value,
                    self.reference_slider_2.value,
                    self.reference_slider_3.value,
                ]
            ).to(device),
            self.seq_len,
            21,
        )
        p = np.exp(log_p)
        df = self.update_datagrid(p)
        df = df.iloc[:, self.data_grid_min : self.data_grid_max]
        self.datagrid.data = df

        with self.graph_panel:
            clear_output(wait=True)
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            # fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
            ax.scatter(
                self.latent_space[:, 0],
                self.latent_space[:, 1],
                self.latent_space[:, 2],
            )
            ax.scatter(
                self.reference_slider_1.value,
                self.reference_slider_2.value,
                self.reference_slider_3.value,
                marker="*",
                s=100,
                c="r",
            )

            ax.set_xlabel("Z1")
            ax.set_ylabel("Z2")
            ax.set_zlabel("Z3", rotation=90, labelpad=0.5)

            plt.show()


demo(
    df,
    latent_space=latent,
    seq_len=len(extants["sequence"][0]),
    data_grid_min=0,
    data_grid_max=281,
)

# -

MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL
VTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV
NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD
HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK

demo(df, latent_space=latent, seq_len=len(extants["sequence"][0]))

# +
from ipywidgets import (
    interact,
    HBox,
    VBox,
    Output,
    HTML,
    Dropdown,
    Button,
    Layout,
    Label,
    Box,
)
from IPython.display import display, clear_output
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import logomaker
from torch import layout  # Assuming you are using logomaker for sequence logos


class demo:

    def __init__(
        self,
        input_df: pd.DataFrame,
        latent_space,
        seq_len,
        data_grid_min,
        data_grid_max,
    ):

        self.data_grid_min = data_grid_min
        self.data_grid_max = data_grid_max

        self.seq_len = seq_len
        self.conditional_expression = Expr(renderer_function)
        self.default_renderer = TextRenderer(
            background_color=self.conditional_expression, format=".3f"
        )
        input_df = input_df.iloc[:, self.data_grid_min : self.data_grid_max]
        self.datagrid = DataGrid(input_df, default_renderer=self.default_renderer)
        self.latent_space = latent_space

        self.operator_dropdown = Dropdown(options=["<", ">"], value="<")
        self.reference_slider_1 = FloatSlider(
            value=0.5, min=-10, max=10, description="Z1"
        )
        self.reference_slider_2 = FloatSlider(
            value=0.5, min=-10, max=10, description="Z2"
        )
        self.reference_slider_3 = FloatSlider(
            value=0.5, min=-10, max=10, description="Z3"
        )
        self.highlight = FloatSlider(value=0.5, min=0, max=1, description="Highlight")
        self.output_colorpicker = ColorPicker(value="#fc8403")

        self.operator_dropdown.observe(
            lambda change: self.update_colour(
                change,
                conditional_expression=self.conditional_expression,
                operator_dropdown=self.operator_dropdown,
                highlight=self.highlight,
                output_colorpicker=self.output_colorpicker,
            ),
            "value",
        )

        self.reference_slider_1.observe(
            lambda change: self.update_probs(change, datagrid=self.datagrid), "value"
        )
        self.reference_slider_2.observe(
            lambda change: self.update_probs(change, datagrid=self.datagrid), "value"
        )
        self.reference_slider_3.observe(
            lambda change: self.update_probs(change, datagrid=self.datagrid), "value"
        )

        self.highlight.observe(
            lambda change: self.update_colour(
                change,
                conditional_expression=self.conditional_expression,
                operator_dropdown=self.operator_dropdown,
                highlight=self.highlight,
                output_colorpicker=self.output_colorpicker,
            ),
            "value",
        )

        self.output_colorpicker.observe(
            lambda change: self.update_colour(
                change,
                conditional_expression=self.conditional_expression,
                operator_dropdown=self.operator_dropdown,
                highlight=self.highlight,
                output_colorpicker=self.output_colorpicker,
            ),
            "value",
        )

        self.button_panel = Output(layout=Layout(width="800px"))
        self.prob_panel = Output(layout=Layout(width="800px"))
        self.graph_panel = Output(layout=Layout(width="800px"))

        self.sequnce_panel = Output(layout=Layout(width="800px"))

        # # data grid setup
        datagrid_controls = VBox(
            [
                HBox(
                    [
                        self.reference_slider_1,
                        self.reference_slider_2,
                        self.reference_slider_3,
                    ]
                ),
                HBox([self.operator_dropdown, self.highlight, self.output_colorpicker]),
            ]
        )

        button_box = VBox([self.prob_panel, self.button_panel, datagrid_controls])

        # scene = VBox([VBox([button_box, self.graph_panel]),
        #               self.datagrid_output, self.sequnce_panel])

        scene = VBox(
            [HBox([self.graph_panel, button_box]), self.datagrid, self.sequnce_panel]
        )

        display(scene)

        with self.button_panel:
            self.btn = Button(description="Update profile")
            self.btn.on_click(self.generate_logo)

            self.start = widgets.BoundedIntText(
                value=self.data_grid_min,
                min=0,
                max=self.seq_len,
                step=1,
                description="Start:",
            )
            self.end = widgets.BoundedIntText(
                value=self.data_grid_max,
                min=0,
                max=self.seq_len,
                step=1,
                description="End:",
            )

            display(HBox([self.btn, self.start, self.end]))

        with self.graph_panel:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                self.latent_space[:, 0],
                self.latent_space[:, 1],
                self.latent_space[:, 2],
            )
            ax.scatter(
                self.reference_slider_1.value,
                self.reference_slider_2.value,
                self.reference_slider_3.value,
                marker="*",
                s=100,
                c="r",
            )
            ax.set_xlabel("Z1")
            ax.set_ylabel("Z2")
            ax.set_zlabel("Z3", rotation=90)

            plt.show()

        with self.sequnce_panel:

            self.seq_input = widgets.Textarea(
                value="",
                placeholder="Enter sequence",
                description="Input:",
                disabled=False,
            )

            self.seq_btn = Button(description="Update")
            self.seq_btn.on_click(self.update_sequence)

            display(HBox([self.seq_input, self.seq_btn]))

        with self.prob_panel:
            clear_output()
            # print(self.datagrid.data.T)
            # print(self.datagrid.data)
            logo_df = self.datagrid.data.iloc[
                :,
                self.start.value
                - self.data_grid_min : self.end.value
                - self.data_grid_min,
            ].T
            # logo_df = self.datagrid.data.T.iloc[self.start.value:self.end.value, :]
            # logo_df = self.datagrid.data.T.iloc[self.start.value - self.data_grid_min:self.end.value - self.data_grid_max, :]
            logo_df = logo_df.rename_axis("pos")

            logo = logomaker.Logo(
                logo_df,
                color_scheme="skylign_protein",
                font_name="Arial Rounded MT Bold",
                show_spines=False,
            )
            plt.xlabel("Sequence index")
            plt.ylabel("Probability")

            plt.show()

    def generate_logo(self, x):

        with self.prob_panel:
            clear_output()
            # print(self.datagrid.data.T)
            # print(self.datagrid.data)
            logo_df = self.datagrid.data.iloc[
                :,
                self.start.value
                - self.data_grid_min : self.end.value
                - self.data_grid_min,
            ].T
            # logo_df = self.datagrid.data.T.iloc[self.start.value:self.end.value, :]
            # logo_df = self.datagrid.data.T.iloc[self.start.value - self.data_grid_min:self.end.value - self.data_grid_max, :]
            logo_df = logo_df.rename_axis("pos")

            logo = logomaker.Logo(
                logo_df,
                color_scheme="skylign_protein",
                font_name="Arial Rounded MT Bold",
                show_spines=False,
            )
            plt.xlabel("Sequence index")
            plt.ylabel("Probability")
            plt.show()

        return

    def update_sequence(self, x):

        cleaned = self.seq_input.value.replace("\n", "")
        df = pd.DataFrame({"id": ["new"], "sequence": [cleaned]})
        encodings = df["sequence"].apply(st.seq_to_one_hot)

        dataset = MSA_Dataset(encodings, np.arange(df.shape[0]), df["id"], device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for x, _, id in loader:
                x = torch.flatten(x, start_dim=1)
                mu, sigma = model.encoder(x)
                eps = torch.randn_like(mu)
                z = mu + sigma * eps
                z = z.detach().cpu().numpy()[0]

        self.reference_slider_1.value = z[0]
        self.reference_slider_2.value = z[1]
        self.reference_slider_3.value = z[2]

        log_p = model.latent_to_log_p(
            torch.Tensor(
                [
                    self.reference_slider_1.value,
                    self.reference_slider_2.value,
                    self.reference_slider_3.value,
                ]
            ).to(device),
            self.seq_len,
            21,
        )
        p = np.exp(log_p)
        df = self.update_datagrid(p)
        df = df.iloc[:, self.data_grid_min : self.data_grid_max]
        self.datagrid.data = df

        with self.graph_panel:
            clear_output(wait=True)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
            # fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
            ax.scatter(
                self.latent_space[:, 0],
                self.latent_space[:, 1],
                self.latent_space[:, 2],
            )
            ax.scatter(
                self.reference_slider_1.value,
                self.reference_slider_2.value,
                self.reference_slider_3.value,
                marker="*",
                s=100,
                c="r",
            )

            ax.set_xlabel("Z1")
            ax.set_ylabel("Z2")
            ax.set_zlabel("Z3", rotation=90, labelpad=0.5)

            plt.show()

    # Function to update the expression and DataGrid
    def update_colour(self, *args, **kwargs):

        kwargs["conditional_expression"].value = (
            "'{color}' if cell.value {operator} {highlight} else default_value".format(
                operator=kwargs["operator_dropdown"].value,
                highlight=kwargs["highlight"].value,
                color=kwargs["output_colorpicker"].value,
            )
        )

    def update_datagrid(self, sequence_log_p: np.ndarray):

        to_plot = sequence_log_p[0, :, :]

        to_plot = to_plot.T

        data = {"data": [], "schema": {}}

        data["data"] = to_plot.tolist()

        data["schema"]["fields"] = [
            {"name": c, type: "number"} for c in st.GAPPY_PROTEIN_ALPHABET
        ]

        return pd.DataFrame(
            data["data"], index=st.GAPPY_PROTEIN_ALPHABET
        )  # , new_logo_data

    def update_probs(self, *args, **kwargs):

        log_p = model.latent_to_log_p(
            torch.Tensor(
                [
                    self.reference_slider_1.value,
                    self.reference_slider_2.value,
                    self.reference_slider_3.value,
                ]
            ).to(device),
            self.seq_len,
            21,
        )
        p = np.exp(log_p)
        df = self.update_datagrid(p)
        df = df.iloc[:, self.data_grid_min : self.data_grid_max]
        self.datagrid.data = df

        with self.graph_panel:
            clear_output(wait=True)
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            # fig, (ax) = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": "3d"})
            ax.scatter(
                self.latent_space[:, 0],
                self.latent_space[:, 1],
                self.latent_space[:, 2],
            )
            ax.scatter(
                self.reference_slider_1.value,
                self.reference_slider_2.value,
                self.reference_slider_3.value,
                marker="*",
                s=100,
                c="r",
            )

            ax.set_xlabel("Z1")
            ax.set_ylabel("Z2")
            ax.set_zlabel("Z3", rotation=90)

            plt.show()


demo(
    df,
    latent_space=latent,
    seq_len=len(extants["sequence"][0]),
    data_grid_min=260,
    data_grid_max=281,
)

# -
