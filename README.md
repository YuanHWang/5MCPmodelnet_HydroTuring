# 5MCPModelNet HydroTuring

This repository contains five MCP-based hydrologic models used in the **HydroTuring** experiments.

The models were developed across three studies by Wang and Gupta (2024, 2025).  
For all three studies, the **Leaf River catchment** was used as the primary experimental test bed for model development, evaluation, and interpretation.

---

## MCP Models

The five models included in this repository are:

1. **MCPModel1:** $MC\{O_{\sigma}L_{\sigma}^{con}\}$
2. **MCPModel2:** $MC\{O_{\sigma}L_{\sigma}^{con}M_{\sigma r}^{R}\}$
3. **MCPModel3:** $MA_{5}$
4. **MCPModel4:** $MN_{SALO}^{DS}(5)$
5. **MCPModel5:** $MN_{SALO}^{DI}(3)$

These five models represent progressively different MCP-based representations of catchment-scale rainfall-runoff dynamics developed in our previous studies.

---

## Model–Paper Mapping

| Model | Model notation | Original study |
|---|---|---|
| MCPModel1 | $MC\{O_{\sigma}L_{\sigma}^{con}\}$ | Wang & Gupta (2024a) |
| MCPModel2 | $MC\{O_{\sigma}L_{\sigma}^{con}M_{\sigma r}^{R}\}$ | Wang & Gupta (2024a) |
| MCPModel3 | $MA_{5}$ | Wang & Gupta (2024b) |
| MCPModel4 | $MN_{SALO}^{DS}(5)$ | Wang & Gupta (2025) |
| MCPModel5 | $MN_{SALO}^{DI}(3)$ | Wang & Gupta (2025) |

### MCPModel1–2

These models were introduced in:

**Wang, Y.H. and Gupta, H.V. (2024a).**  
*A mass-conserving-perceptron for machine-learning-based modeling of geoscientific systems.*  
**Water Resources Research, 60**(4), e2023WR036461.

This study introduced the Mass-Conserving Perceptron (MCP) as a differentiable, mass-conserving architectural unit for modeling dynamical geoscientific systems.

### MCPModel3

This model was developed in:

**Wang, Y.H. and Gupta, H.V. (2024b).**  
*Towards interpretable physical-conceptual catchment-scale hydrological modeling using the mass-conserving-perceptron.*  
**Water Resources Research, 60**(10), e2024WR037224.

This study explored how MCP-based architectures can be assembled to construct interpretable physical-conceptual hydrologic models.

### MCPModel4–5

These models were developed in:

**Wang, Y.H. and Gupta, H.V. (2025).**  
*Using machine learning to discover parsimonious and physically-interpretable representations of catchment-scale rainfall-runoff dynamics.*  
**Water Resources Research, 61**(12), e2025WR040178.

This study used MCP-based model networks to investigate parsimonious and physically interpretable representations of catchment-scale rainfall-runoff dynamics.

---

## Experimental Catchment

The **Leaf River catchment** serves as the primary experimental catchment underlying the development and evaluation of the five model structures included in this repository.

The HydroTuring experiments use these previously developed MCP architectures as candidate hydrologic model structures for systematic comparison, evaluation, and interpretation.

---

## Environment Setup

The code was tested using **Python 3.10**.

A Conda environment file and a pip requirements file are provided in this repository:

- `environment.yml`
- `requirements.txt`

### Option 1: Conda

Create the environment using:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate hydroturing
```

### Option 2: pip

Alternatively, install the required Python packages using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- PyTorch
- NumPy
- Pandas
- scikit-learn
- torchvision
- tqdm

---

## Repository Structure

A simplified structure of the repository is:

```text
5MCPmodelnet_HydroTuring/
│
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── MCPModel1/
├── MCPModel2/
├── MCPModel3/
├── MCPModel4/
├── MCPModel5/
│
├── MCPBRNN_lib_tools/
└── data/
```

The exact directory structure may vary slightly among model implementations because the five architectures were developed across different stages of the MCP studies.

---

## References

1. **Wang, Y.H. and Gupta, H.V. (2024a).**  
   A mass-conserving-perceptron for machine-learning-based modeling of geoscientific systems.  
   *Water Resources Research, 60*(4), e2023WR036461.

2. **Wang, Y.H. and Gupta, H.V. (2024b).**  
   Towards interpretable physical-conceptual catchment-scale hydrological modeling using the mass-conserving-perceptron.  
   *Water Resources Research, 60*(10), e2024WR037224.

3. **Wang, Y.H. and Gupta, H.V. (2025).**  
   Using machine learning to discover parsimonious and physically-interpretable representations of catchment-scale rainfall-runoff dynamics.  
   *Water Resources Research, 61*(12), e2025WR040178.
