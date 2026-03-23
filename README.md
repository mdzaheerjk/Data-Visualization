# Matplotlib & Seaborn — Complete Reference Notes
## With Titanic Dataset Examples

---

## Setup & Imports

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np

# Load Titanic dataset (built into seaborn)
titanic = sns.load_dataset("titanic")
# Columns: survived, pclass, sex, age, sibsp, parch, fare, embarked,
#          class, who, adult_male, deck, embark_town, alive, alone

# Quick peek
print(titanic.shape)        # (891, 15)
print(titanic.dtypes)
print(titanic.head())
print(titanic.isnull().sum())
```

---

## Matplotlib Fundamentals

### The Object Model

```
Figure  ← top-level container (the whole image)
  └── Axes  ← individual plot area (has x-axis, y-axis, title, etc.)
        ├── Axis  ← x or y axis
        ├── Artist  ← everything drawn (lines, patches, text, images)
        └── Legend, Colorbar, etc.
```

### Two interfaces

```python
# 1. Pyplot interface (implicit, quick plots)
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

# 2. Object-oriented interface (explicit, recommended for complex plots)
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

### Figure & Axes creation

```python
# Single plot
fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(10, 6))    # width × height in inches
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)   # dots per inch

# Multiple subplots
fig, axes = plt.subplots(2, 3)             # 2 rows, 3 cols → axes.shape = (2,3)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig, (ax1, ax2) = plt.subplots(1, 2)      # unpack directly
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# Shared axes
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
fig, axes = plt.subplots(1, 2, sharey='row')
fig, axes = plt.subplots(2, 1, sharex='col')

# Figure only, add axes manually
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)           # 1 row, 1 col, plot 1
ax = fig.add_subplot(2, 2, 1)       # 2×2 grid, position 1
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])   # [left, bottom, width, height] 0-1
```

---

## Core Plot Types — Matplotlib

### Line Plot

```python
# Basic survival rate by age
survived_by_age = titanic.groupby("age")["survived"].mean().dropna()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(survived_by_age.index, survived_by_age.values)

# Styling options
ax.plot(survived_by_age.index, survived_by_age.values,
    color="#E63946",          # hex color
    linewidth=2,              # lw=2
    linestyle="--",           # ls="--"  options: '-', '--', '-.', ':'
    marker="o",               # marker style
    markersize=5,             # ms=5
    markerfacecolor="white",
    markeredgecolor="#E63946",
    markeredgewidth=1.5,
    alpha=0.8,
    label="Survival Rate"
)

# Multiple lines
for cls, grp in titanic.groupby("pclass"):
    rate = grp.groupby("age")["survived"].mean().dropna().rolling(5).mean()
    ax.plot(rate.index, rate.values, label=f"Class {cls}", linewidth=2)

ax.set_title("Survival Rate by Age", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Survival Rate", fontsize=12)
ax.legend(title="Passenger Class", fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(0, 80)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.show()
```

### Marker styles reference

```python
markers = ['o','s','^','v','<','>','D','P','*','X','h','8','p','+','x','|','_','.']
# o=circle  s=square  ^=up-triangle  v=down-triangle  D=diamond
# P=plus(filled)  *=star  X=x(filled)  h=hexagon  .=point
```

### Line style reference

```python
linestyles = ['-', '--', '-.', ':']
# solid, dashed, dash-dot, dotted
```

### Bar Chart

```python
# Survival count by class
class_survival = titanic.groupby("pclass")["survived"].agg(["sum","count"])
class_survival["not_survived"] = class_survival["count"] - class_survival["sum"]

fig, ax = plt.subplots(figsize=(8, 6))

x = np.array([1, 2, 3])
width = 0.35

bars1 = ax.bar(x - width/2, class_survival["sum"],
               width=width, label="Survived",
               color="#2A9D8F", edgecolor="white", linewidth=0.8)
bars2 = ax.bar(x + width/2, class_survival["not_survived"],
               width=width, label="Did Not Survive",
               color="#E76F51", edgecolor="white", linewidth=0.8)

# Add value labels on bars
ax.bar_label(bars1, padding=3, fontsize=10, fontweight="bold")
ax.bar_label(bars2, padding=3, fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
ax.set_title("Survival Count by Passenger Class", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Passengers")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
```

### Horizontal Bar Chart

```python
embarked_survival = titanic.groupby("embark_town")["survived"].mean().dropna().sort_values()

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#E76F51" if v < 0.4 else "#2A9D8F" for v in embarked_survival.values]
bars = ax.barh(embarked_survival.index, embarked_survival.values,
               color=colors, edgecolor="white", height=0.5)
ax.bar_label(bars, fmt="%.1%", padding=5)
ax.set_xlabel("Survival Rate")
ax.set_title("Survival Rate by Embarkation Town")
ax.set_xlim(0, 0.7)
plt.tight_layout()
plt.show()
```

### Histogram

```python
fig, ax = plt.subplots(figsize=(10, 5))

survived   = titanic[titanic["survived"] == 1]["age"].dropna()
not_survived = titanic[titanic["survived"] == 0]["age"].dropna()

ax.hist(not_survived, bins=30, alpha=0.6, color="#E76F51",
        label="Did Not Survive", edgecolor="white")
ax.hist(survived,   bins=30, alpha=0.6, color="#2A9D8F",
        label="Survived",       edgecolor="white")

# Density histogram
ax.hist(survived, bins=30, density=True, alpha=0.6, color="#2A9D8F")

# Cumulative
ax.hist(survived, bins=30, cumulative=True, density=True, histtype="step",
        color="#2A9D8F", linewidth=2)

ax.set_title("Age Distribution by Survival Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend(fontsize=11)
ax.axvline(survived.mean(), color="#2A9D8F", linestyle="--", linewidth=1.5,
           label=f"Survived mean: {survived.mean():.1f}")
ax.axvline(not_survived.mean(), color="#E76F51", linestyle="--", linewidth=1.5,
           label=f"Not survived mean: {not_survived.mean():.1f}")
plt.tight_layout()
plt.show()
```

### Scatter Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))

colors = titanic["survived"].map({0: "#E76F51", 1: "#2A9D8F"})
sizes  = titanic["fare"].fillna(0) * 0.5 + 20

scatter = ax.scatter(
    titanic["age"].fillna(titanic["age"].median()),
    titanic["fare"],
    c=colors,
    s=sizes,
    alpha=0.5,
    edgecolors="white",
    linewidth=0.5
)

# Custom legend
patch_s = mpatches.Patch(color="#2A9D8F", label="Survived")
patch_n = mpatches.Patch(color="#E76F51", label="Not Survived")
ax.legend(handles=[patch_s, patch_n], title="Outcome")

ax.set_title("Age vs Fare (size = fare)", fontsize=14, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Fare (£)")
ax.set_yscale("log")      # log scale for skewed fare
plt.tight_layout()
plt.show()
```

### Pie Chart

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Overall survival
vals = titanic["survived"].value_counts()
axes[0].pie(vals, labels=["Did Not Survive", "Survived"],
            colors=["#E76F51", "#2A9D8F"],
            autopct="%1.1f%%", startangle=90,
            explode=[0, 0.05],
            shadow=True,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Overall Survival")

# By class
class_counts = titanic["pclass"].value_counts().sort_index()
axes[1].pie(class_counts, labels=["1st", "2nd", "3rd"],
            autopct="%1.1f%%", startangle=90,
            colors=["#264653", "#2A9D8F", "#E9C46A"])
axes[1].set_title("Passengers by Class")

plt.suptitle("Titanic Overview", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
```

### Box Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fare distribution by class
groups = [titanic[titanic["pclass"] == c]["fare"].dropna() for c in [1,2,3]]
bp = axes[0].boxplot(groups,
    labels=["1st Class", "2nd Class", "3rd Class"],
    patch_artist=True,                 # filled boxes
    notch=True,                        # confidence interval notch
    vert=True,
    showfliers=True,                   # show outliers
    showmeans=True,
    meanprops={"marker":"D","markerfacecolor":"white","markersize":8}
)

colors = ["#264653", "#2A9D8F", "#E9C46A"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0].set_title("Fare by Passenger Class")
axes[0].set_ylabel("Fare (£)")

# Age distribution by sex and survival
for i, (survived, grp) in enumerate(titanic.groupby("survived")):
    label = "Survived" if survived else "Not Survived"
    color = "#2A9D8F" if survived else "#E76F51"
    male = grp[grp["sex"] == "male"]["age"].dropna()
    female = grp[grp["sex"] == "female"]["age"].dropna()
    pos = [i*2.5, i*2.5 + 1]
    bp2 = axes[1].boxplot([male, female], positions=pos, widths=0.7,
                          patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

axes[1].set_xticks([0.5, 3.0])
axes[1].set_xticklabels(["Not Survived", "Survived"])
axes[1].set_title("Age by Sex and Survival")
axes[1].set_ylabel("Age")

plt.tight_layout()
plt.show()
```

### Error Bar Plot

```python
age_by_class = titanic.groupby("pclass")["age"].agg(["mean","std","count"])
age_by_class["se"] = age_by_class["std"] / np.sqrt(age_by_class["count"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    age_by_class.index,
    age_by_class["mean"],
    yerr=age_by_class["se"] * 1.96,    # 95% CI
    fmt="o-",
    capsize=8,
    capthick=2,
    linewidth=2,
    markersize=10,
    color="#264653",
    ecolor="#2A9D8F",
    label="Mean Age ± 95% CI"
)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
ax.set_title("Mean Age by Class with 95% CI")
ax.set_ylabel("Age (years)")
ax.legend()
plt.tight_layout()
plt.show()
```

### Fill Between

```python
fare_by_class = titanic.groupby(["pclass","age"])["fare"].mean().unstack(0)
fare_smooth = fare_by_class.interpolate().rolling(3).mean().dropna()

fig, ax = plt.subplots(figsize=(11, 5))
ages = fare_smooth.index
ax.plot(ages, fare_smooth[1], color="#264653", label="1st Class", lw=2)
ax.fill_between(ages, fare_smooth[1]*0.8, fare_smooth[1]*1.2,
                color="#264653", alpha=0.15)
ax.plot(ages, fare_smooth[2], color="#2A9D8F", label="2nd Class", lw=2)
ax.fill_between(ages, fare_smooth[2], fare_smooth[3],
                color="#E9C46A", alpha=0.3, label="3rd Class range")
ax.set_title("Average Fare by Age and Class")
ax.set_xlabel("Age"); ax.set_ylabel("Fare (£)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

### Heatmap (Matplotlib)

```python
# Correlation matrix
num_cols = ["survived","pclass","age","sibsp","parch","fare"]
corr = titanic[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=45, ha="right")
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}",
                ha="center", va="center",
                color="black" if abs(corr.iloc[i,j]) < 0.5 else "white",
                fontsize=9, fontweight="bold")

ax.set_title("Correlation Matrix — Titanic Features", fontsize=14, pad=15)
plt.tight_layout(); plt.show()
```

---

## Axes Customisation

### Titles, Labels, Ticks

```python
ax.set_title("Title", fontsize=16, fontweight="bold", color="#264653", pad=20)
ax.set_xlabel("X Label", fontsize=13, labelpad=10)
ax.set_ylabel("Y Label", fontsize=13, labelpad=10)

ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_xlim(left=0)           # only set one bound

ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xticklabels(["0", "20", "40", "60", "80"], fontsize=10, rotation=45)
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.tick_params(axis="both", labelsize=10, length=5, width=1.5, direction="in")
ax.tick_params(axis="x", rotation=45)

# Invert axis
ax.invert_xaxis()
ax.invert_yaxis()

# Secondary axis
ax2 = ax.twinx()              # share x, independent y
ax2 = ax.twiny()              # share y, independent x
ax2.set_ylabel("Second Y", color="red")
ax2.tick_params(axis="y", colors="red")
```

### Formatters & Locators (ticker)

```python
from matplotlib import ticker

# Formatters
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))    # 0.5 → "50%"
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"£{x:,.0f}"))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

# Locators
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))    # tick every 10
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))     # minor every 5
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))         # max 5 ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(6))       # exactly 6 ticks
ax.xaxis.set_major_locator(ticker.LogLocator(base=10))    # log scale
```

### Grid & Spines

```python
ax.grid(True)
ax.grid(True, which="major", linestyle="--", alpha=0.5, color="gray")
ax.grid(True, which="minor", linestyle=":", alpha=0.3)
ax.grid(True, axis="y")             # only horizontal grid lines
ax.minorticks_on()

# Spines (the four border lines)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_color("#264653")
ax.spines["left"].set_position(("outward", 10))   # move spine outward
```

### Legend

```python
ax.legend()
ax.legend(loc="upper right")     # 'best','upper left','lower center', etc.
ax.legend(loc=(0.7, 0.5))        # manual position (0-1 coords)
ax.legend(
    title="Legend Title",
    title_fontsize=11,
    fontsize=10,
    frameon=True,
    framealpha=0.9,
    edgecolor="gray",
    fancybox=True,
    shadow=True,
    ncol=2,                       # columns in legend
    bbox_to_anchor=(1.05, 1),     # place outside plot
    borderpad=1,
    labelspacing=0.5
)

# Custom legend entries
handle1 = mpatches.Patch(color="#2A9D8F", label="Survived")
handle2 = mlines.Line2D([], [], color="#E76F51", marker="o", label="Not Survived")
ax.legend(handles=[handle1, handle2])
```

### Annotations & Text

```python
# Simple text
ax.text(30, 0.8, "Peak survival rate", fontsize=10, color="gray",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

# Arrow annotation
ax.annotate(
    "Children survive\nmore often",
    xy=(5, 0.7),           # point to
    xytext=(20, 0.9),      # text position
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8)
)

# Horizontal / vertical lines
ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="50% line")
ax.axvline(x=30, color="blue", linestyle=":", linewidth=1)
ax.axhspan(ymin=0.3, ymax=0.7, facecolor="yellow", alpha=0.1)  # horizontal band
ax.axvspan(xmin=0, xmax=10, facecolor="green", alpha=0.1)       # vertical band
```

---

## Figure-Level Customisation

```python
fig.suptitle("Titanic Data Analysis", fontsize=18, fontweight="bold", y=1.02)
fig.text(0.5, -0.02, "Source: Seaborn Titanic dataset",
         ha="center", fontsize=9, color="gray")

plt.tight_layout()                         # auto-adjust spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])    # leave room for suptitle
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                    hspace=0.4, wspace=0.3)
```

---

## Advanced Layouts

### GridSpec

```python
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])      # top row spans all columns
ax2 = fig.add_subplot(gs[1, 0:2])    # middle row, first two columns
ax3 = fig.add_subplot(gs[1, 2])      # middle row, last column
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[2, 2])

# Different size ratios
gs2 = gridspec.GridSpec(2, 2,
    width_ratios=[3, 1],
    height_ratios=[1, 2])
```

### Inset Axes

```python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(survived_by_age.index, survived_by_age.values)

# Add inset
ax_inset = inset_axes(ax, width="35%", height="35%", loc="upper right")
ax_inset.hist(titanic["age"].dropna(), bins=20, color="gray", alpha=0.6)
ax_inset.set_title("Age dist.", fontsize=8)
ax_inset.tick_params(labelsize=7)
```

---

## Styles & Colormaps

### Style sheets

```python
plt.style.use("seaborn-v0_8")          # clean seaborn look
plt.style.use("ggplot")                # R ggplot look
plt.style.use("fivethirtyeight")       # 538 journalism style
plt.style.use("dark_background")       # dark mode
plt.style.use("bmh")                   # Bayesian Methods for Hackers
plt.style.use("tableau-colorblind10")  # accessible colors
plt.style.use("default")               # reset to default

# Temporary style
with plt.style.context("ggplot"):
    plt.plot([1, 2, 3])

# List all
print(plt.style.available)
```

### Colormaps

```python
# Sequential (low to high)
cmaps_seq  = ["viridis","plasma","inferno","magma","cividis",
              "Blues","Greens","Reds","Oranges","Purples","YlOrRd"]

# Diverging (center emphasis)
cmaps_div  = ["RdYlGn","RdBu","seismic","coolwarm","bwr","PiYG","PRGn"]

# Qualitative (categorical)
cmaps_qual = ["tab10","tab20","Set1","Set2","Set3","Pastel1","Paired"]

# Perceptually uniform (scientific)
cmaps_uni  = ["viridis","plasma","cividis"]

# Use in plot
im = ax.imshow(matrix, cmap="viridis")
ax.scatter(x, y, c=z, cmap="RdYlGn", vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Correlation", fontsize=11)
cbar.ax.tick_params(labelsize=9)

# Get N discrete colors from a colormap
colors = plt.cm.Set2(np.linspace(0, 1, 5))

# Reverse colormap
ax.scatter(x, y, c=z, cmap="viridis_r")   # append _r
```

### Named colors

```python
# CSS named colors
"red", "blue", "green", "white", "black", "gray", "orange", "purple"
# Short codes
"r", "g", "b", "c", "m", "y", "k", "w"
# Hex
"#E63946", "#457B9D", "#1D3557"
# RGB tuple (0-1)
(0.2, 0.5, 0.8)
# RGBA
(0.2, 0.5, 0.8, 0.5)
# Grayscale string
"0.75"   # 75% gray
```

---

## Saving Figures

```python
plt.savefig("titanic_plot.png")
plt.savefig("titanic_plot.png", dpi=300, bbox_inches="tight")
plt.savefig("titanic_plot.pdf", bbox_inches="tight")    # vector PDF
plt.savefig("titanic_plot.svg")                          # vector SVG
plt.savefig("titanic_plot.jpg", dpi=150, quality=95)

fig.savefig("plot.png", dpi=300, bbox_inches="tight",
            facecolor="white",         # white background (transparent by default)
            edgecolor="none",
            transparent=False)

# Formats: png, pdf, svg, eps, ps, jpg, tiff, webp
```

---

## Seaborn Overview

Seaborn is built on top of Matplotlib with a high-level API that:
- works natively with DataFrames
- handles grouping/hue/style/size automatically
- has beautiful default themes
- produces statistical plots easily

```python
# Set theme
sns.set_theme()                          # default clean theme
sns.set_theme(style="darkgrid")
sns.set_theme(style="whitegrid")         # most common for publications
sns.set_theme(style="white")
sns.set_theme(style="ticks")
sns.set_theme(context="paper")           # smaller fonts
sns.set_theme(context="notebook")        # default
sns.set_theme(context="talk")            # larger (presentations)
sns.set_theme(context="poster")          # largest

sns.set_palette("husl")                  # change default color palette
sns.set_palette("Set2")
sns.set_palette(["#264653","#2A9D8F","#E9C46A","#F4A261","#E76F51"])

# Reset
sns.reset_defaults()
sns.reset_orig()                         # back to pure matplotlib defaults
```

---

## Seaborn — Relational Plots

### sns.scatterplot

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=titanic,
    x="age", y="fare",
    hue="survived",          # color by column
    style="sex",             # marker shape by column
    size="pclass",           # marker size by column
    sizes=(40, 200),         # size range
    palette={0:"#E76F51", 1:"#2A9D8F"},
    alpha=0.7,
    ax=ax
)
ax.set_title("Age vs Fare by Survival, Sex, and Class")
plt.show()
```

### sns.lineplot

```python
# Survival rate over age — auto handles mean + CI
fig, ax = plt.subplots(figsize=(11, 5))
sns.lineplot(
    data=titanic,
    x="age", y="survived",
    hue="pclass",
    estimator="mean",         # aggregate function
    errorbar="ci",            # 'ci', 'sd', 'se', 'pi', None
    err_style="band",         # 'band' or 'bars'
    n_boot=200,               # bootstrap samples for CI
    palette="Set2",
    linewidth=2,
    ax=ax
)
ax.set_title("Survival Rate by Age and Class (95% CI)", fontsize=14)
ax.set_xlabel("Age"); ax.set_ylabel("Survival Rate")
plt.show()
```

---

## Seaborn — Distribution Plots

### sns.histplot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with KDE
sns.histplot(
    data=titanic, x="age",
    hue="survived",
    bins=30,
    kde=True,                 # overlay KDE curve
    stat="density",           # 'count','frequency','density','probability'
    common_norm=False,        # normalize each group separately
    palette={0:"#E76F51", 1:"#2A9D8F"},
    alpha=0.5,
    ax=axes[0]
)
axes[0].set_title("Age Distribution by Survival")

# 2D histogram
sns.histplot(
    data=titanic.dropna(subset=["age","fare"]),
    x="age", y="fare",
    bins=30,
    cmap="Blues",
    ax=axes[1]
)
axes[1].set_title("2D Histogram: Age vs Fare")
plt.tight_layout(); plt.show()
```

### sns.kdeplot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1D KDE
sns.kdeplot(
    data=titanic, x="age",
    hue="survived",
    fill=True,                # shade under curve
    common_norm=False,
    alpha=0.4,
    bw_adjust=0.8,            # bandwidth adjustment (lower=rougher, higher=smoother)
    cut=0,                    # don't extend beyond data
    palette={0:"#E76F51", 1:"#2A9D8F"},
    ax=axes[0]
)
axes[0].set_title("Age KDE by Survival")

# 2D KDE (bivariate)
subset = titanic.dropna(subset=["age","fare"])
sns.kdeplot(
    data=subset,
    x="age", y="fare",
    levels=10,                # number of contour levels
    fill=True,
    cmap="Blues",
    thresh=0.05,              # clip below this density
    ax=axes[1]
)
axes[1].set_title("2D KDE: Age vs Fare")
plt.tight_layout(); plt.show()
```

### sns.ecdfplot (empirical CDF)

```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.ecdfplot(
    data=titanic, x="fare",
    hue="pclass",
    complementary=False,      # True = survival function (1 - CDF)
    stat="proportion",        # 'proportion' or 'count'
    ax=ax
)
ax.set_title("Empirical CDF of Fare by Class")
ax.set_xscale("log")
plt.show()
```

### sns.rugplot

```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(data=titanic, x="age", hue="survived", fill=True, alpha=0.3, ax=ax)
sns.rugplot(data=titanic, x="age", hue="survived",
            height=0.05, alpha=0.3, ax=ax)
ax.set_title("Age KDE + Rug")
plt.show()
```

---

## Seaborn — Categorical Plots

### sns.boxplot

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=titanic,
    x="pclass", y="age",
    hue="survived",
    palette={0:"#E76F51", 1:"#2A9D8F"},
    width=0.6,
    flierprops={"marker":"o","markerfacecolor":"gray","alpha":0.3},
    medianprops={"color":"white","linewidth":2},
    notch=False,
    order=[1, 2, 3],
    hue_order=[0, 1],
    ax=ax
)
ax.set_title("Age Distribution by Class and Survival")
ax.set_xlabel("Passenger Class"); ax.set_ylabel("Age")
plt.show()
```

### sns.violinplot

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(
    data=titanic,
    x="pclass", y="fare",
    hue="survived",
    split=True,              # split violin by hue (only for 2 hue values)
    inner="quart",           # 'box','quart','point','stick', None
    palette={0:"#E76F51", 1:"#2A9D8F"},
    bw_adjust=0.8,
    scale="width",           # 'area','count','width'
    cut=0,
    density_norm="width",
    ax=ax
)
ax.set_yscale("log")
ax.set_title("Fare Distribution by Class and Survival (Split Violin)")
plt.show()
```

### sns.boxenplot (letter-value plot)

```python
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxenplot(
    data=titanic, x="pclass", y="fare",
    hue="survived",
    palette={0:"#E76F51", 1:"#2A9D8F"},
    scale="linear",
    k_depth="tukey",          # 'proportion','tukey','trustworthy'
    ax=ax
)
ax.set_title("Fare Distribution — Boxen Plot (Better for Large Datasets)")
plt.show()
```

### sns.stripplot & swarmplot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Strip plot (jittered points)
sns.stripplot(
    data=titanic, x="pclass", y="age",
    hue="survived",
    dodge=True,              # separate by hue
    jitter=0.2,
    alpha=0.4, size=4,
    palette={0:"#E76F51", 1:"#2A9D8F"},
    ax=axes[0]
)
axes[0].set_title("Strip Plot: Age by Class and Survival")

# Swarm plot (no overlap — exact positions)
sns.swarmplot(
    data=titanic[titanic["pclass"] == 1],  # smaller subset
    x="sex", y="age",
    hue="survived",
    palette={0:"#E76F51", 1:"#2A9D8F"},
    size=4,
    ax=axes[1]
)
axes[1].set_title("Swarm Plot: Age by Sex (1st Class)")
plt.tight_layout(); plt.show()
```

### sns.barplot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean survival rate with CI
sns.barplot(
    data=titanic,
    x="pclass", y="survived",
    hue="sex",
    palette="Set2",
    estimator="mean",
    errorbar="ci",
    capsize=0.1,
    order=[1, 2, 3],
    ax=axes[0]
)
axes[0].set_title("Survival Rate by Class and Sex (Mean ± 95% CI)")
axes[0].set_ylabel("Survival Rate")

# Count
sns.barplot(
    data=titanic,
    x="embark_town", y="fare",
    hue="pclass",
    estimator="median",
    errorbar="sd",
    palette="husl",
    ax=axes[1]
)
axes[1].set_title("Median Fare by Embarkation and Class")
plt.tight_layout(); plt.show()
```

### sns.countplot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(
    data=titanic, x="pclass",
    hue="survived",
    palette={0:"#E76F51", 1:"#2A9D8F"},
    order=[1, 2, 3],
    stat="count",             # 'count','percent','proportion','probability'
    ax=axes[0]
)
axes[0].set_title("Passenger Count by Class and Survival")

# Horizontal
sns.countplot(
    data=titanic, y="embark_town",
    hue="pclass",
    palette="Set2",
    order=titanic["embark_town"].value_counts().index,
    ax=axes[1]
)
axes[1].set_title("Embarkation Count by Class")
plt.tight_layout(); plt.show()
```

### sns.pointplot

```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.pointplot(
    data=titanic,
    x="pclass", y="survived",
    hue="sex",
    palette="Set2",
    estimator="mean",
    errorbar="ci",
    capsize=0.15,
    markers=["o","s"],
    linestyles=["-","--"],
    dodge=0.3,                # separate overlapping points
    ax=ax
)
ax.set_title("Survival Rate by Class and Sex")
ax.set_ylabel("Survival Rate")
plt.show()
```

---

## Seaborn — Matrix Plots

### sns.heatmap

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Correlation heatmap
corr = titanic[["survived","pclass","age","sibsp","parch","fare"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))    # mask upper triangle

sns.heatmap(
    corr,
    ax=axes[0],
    mask=mask,
    annot=True,              # show values in cells
    fmt=".2f",               # format for values
    cmap="RdYlGn",
    vmin=-1, vmax=1,
    center=0,                # center colormap at 0
    linewidths=0.5,          # cell borders
    linecolor="white",
    square=True,             # force square cells
    cbar_kws={"shrink":0.8, "label":"Correlation"},
    annot_kws={"size":10, "weight":"bold"}
)
axes[0].set_title("Feature Correlation (Lower Triangle)")

# Pivot heatmap
pivot = titanic.groupby(["pclass","sex"])["survived"].mean().unstack()
sns.heatmap(
    pivot,
    ax=axes[1],
    annot=True, fmt=".1%",
    cmap="RdYlGn",
    vmin=0, vmax=1,
    linewidths=2, linecolor="white",
    cbar_kws={"format":"{:.0%}"}
)
axes[1].set_title("Survival Rate by Class & Sex")
plt.tight_layout(); plt.show()
```

### sns.clustermap

```python
# Hierarchically clustered heatmap
subset = titanic[["pclass","age","sibsp","parch","fare","survived"]].dropna()
subset_norm = (subset - subset.mean()) / subset.std()

g = sns.clustermap(
    subset_norm.T,
    method="ward",           # linkage: 'single','complete','average','ward'
    metric="euclidean",
    cmap="vlag",
    figsize=(14, 6),
    z_score=1,               # normalise rows (0=cols, 1=rows)
    col_cluster=True,        # cluster columns (passengers)
    row_cluster=True,        # cluster rows (features)
    annot=False,
    yticklabels=True,
    xticklabels=False,
    cbar_pos=(0.02, 0.8, 0.03, 0.18)
)
g.fig.suptitle("Clustered Feature Heatmap — Titanic", y=1.02)
plt.show()
```

---

## Seaborn — Figure-Level Functions

These return a `FacetGrid` or `PairGrid` object (not an Axes).

### sns.relplot (scatter or line across facets)

```python
g = sns.relplot(
    data=titanic.dropna(subset=["age","fare","deck"]),
    x="age", y="fare",
    hue="survived",
    style="sex",
    size="pclass",
    sizes=(30, 200),
    col="pclass",             # one column per class
    col_wrap=3,               # wrap at 3 columns
    kind="scatter",           # or "line"
    palette={0:"#E76F51", 1:"#2A9D8F"},
    alpha=0.6,
    height=4, aspect=1.2
)
g.set_titles("Class {col_name}")
g.set_axis_labels("Age", "Fare")
g.figure.suptitle("Age vs Fare by Class", y=1.02)
plt.show()
```

### sns.displot (distribution across facets)

```python
g = sns.displot(
    data=titanic,
    x="age",
    hue="survived",
    col="pclass",
    kind="kde",               # 'hist','kde','ecdf'
    fill=True, alpha=0.4,
    common_norm=False,
    palette={0:"#E76F51", 1:"#2A9D8F"},
    height=4, aspect=1.2
)
g.set_titles("Class {col_name}")
g.set_axis_labels("Age", "Density")
g.figure.suptitle("Age Distribution by Class and Survival", y=1.02)
plt.show()
```

### sns.catplot (categorical across facets)

```python
g = sns.catplot(
    data=titanic,
    x="sex", y="survived",
    col="pclass",
    kind="bar",               # 'strip','swarm','box','violin','bar','count','point'
    palette="Set2",
    estimator="mean",
    errorbar="ci",
    height=5, aspect=0.7
)
g.set_axis_labels("Sex", "Survival Rate")
g.set_titles("Class {col_name}")
g.set(ylim=(0, 1))
g.figure.suptitle("Survival Rate by Sex and Class", y=1.02)
plt.show()
```

### sns.pairplot

```python
# Pairwise relationships of all numeric columns
g = sns.pairplot(
    data=titanic[["age","fare","pclass","sibsp","parch","survived"]].dropna(),
    hue="survived",
    palette={0:"#E76F51", 1:"#2A9D8F"},
    diag_kind="kde",          # diagonal: 'auto','hist','kde', None
    plot_kws={"alpha":0.4, "s":20},
    diag_kws={"fill":True, "common_norm":False},
    corner=False              # True = lower triangle only
)
g.figure.suptitle("Pairplot — Titanic Numeric Features", y=1.02)
plt.show()
```

### sns.jointplot

```python
# Bivariate + marginal distributions
g = sns.jointplot(
    data=titanic.dropna(subset=["age","fare"]),
    x="age", y="fare",
    hue="survived",
    kind="scatter",           # 'scatter','kde','hist','hex','reg','resid'
    palette={0:"#E76F51", 1:"#2A9D8F"},
    alpha=0.5,
    marginal_kws={"fill":True, "alpha":0.4},
    height=8
)
g.figure.suptitle("Age vs Fare — Joint Distribution", y=1.02)
plt.show()

# KDE version
g = sns.jointplot(
    data=titanic.dropna(subset=["age","fare"]),
    x="age", y="fare",
    kind="kde",
    fill=True, thresh=0.05,
    levels=10, cmap="Blues",
    marginal_kws={"fill":True}
)
```

### sns.FacetGrid (manual)

```python
g = sns.FacetGrid(
    titanic, col="pclass", row="sex",
    hue="survived",
    palette={0:"#E76F51", 1:"#2A9D8F"},
    height=3, aspect=1.2,
    sharey=True, sharex=True,
    margin_titles=True
)
g.map(sns.kdeplot, "age", fill=True, alpha=0.4, common_norm=False)
g.map(sns.rugplot, "age", alpha=0.3)
g.add_legend()
g.set_titles(row_template="{row_name}", col_template="Class {col_name}")
g.set_axis_labels("Age", "Density")
g.figure.suptitle("Age Distribution by Class, Sex and Survival", y=1.02)
plt.show()
```

---

## Regression & Statistical Plots

### sns.regplot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.regplot(
    data=titanic,
    x="age", y="fare",
    scatter_kws={"alpha":0.3, "s":20},
    line_kws={"color":"red", "linewidth":2},
    ci=95,                    # confidence interval
    order=1,                  # polynomial order (1=linear, 2=quadratic)
    robust=False,             # robust regression
    logx=False,               # log-transform x
    lowess=False,             # locally weighted regression
    ax=axes[0]
)
axes[0].set_title("Age vs Fare — Linear Regression")

# Logistic regression for binary outcome
sns.regplot(
    data=titanic.dropna(subset=["age"]),
    x="age", y="survived",
    logistic=True,            # logistic regression
    scatter_kws={"alpha":0.2, "s":15},
    line_kws={"color":"#264653", "linewidth":2.5},
    ci=95,
    ax=axes[1]
)
axes[1].set_title("Age vs Survival Probability — Logistic Regression")
axes[1].set_ylabel("P(Survived)")
plt.tight_layout(); plt.show()
```

### sns.lmplot (regplot with facets)

```python
g = sns.lmplot(
    data=titanic.dropna(subset=["age","fare"]),
    x="age", y="fare",
    hue="sex",
    col="pclass",
    logistic=False,
    ci=90,
    scatter_kws={"alpha":0.3, "s":20},
    line_kws={"linewidth":2},
    palette="Set2",
    height=5, aspect=1
)
g.set_titles("Class {col_name}")
g.figure.suptitle("Age vs Fare Regression by Class and Sex", y=1.02)
plt.show()
```

### sns.residplot

```python
fig, ax = plt.subplots(figsize=(9, 5))
sns.residplot(
    data=titanic.dropna(subset=["age","fare"]),
    x="age", y="fare",
    lowess=True,
    scatter_kws={"alpha":0.3},
    line_kws={"color":"red"},
    ax=ax
)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Residuals: Age vs Fare")
plt.show()
```

---

## Complete Multi-Panel Dashboard

```python
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Titanic Dataset — Complete Analysis Dashboard",
             fontsize=20, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Survival by class (bar)
ax1 = fig.add_subplot(gs[0, 0])
sns.countplot(data=titanic, x="pclass", hue="survived",
              palette={0:"#E76F51", 1:"#2A9D8F"}, ax=ax1)
ax1.set_title("Passengers by Class & Survival", fontweight="bold")
ax1.set_xlabel("Class"); ax1.set_ylabel("Count")
ax1.legend(["Not Survived","Survived"], fontsize=8)

# 2. Age KDE by survival
ax2 = fig.add_subplot(gs[0, 1])
sns.kdeplot(data=titanic, x="age", hue="survived",
            fill=True, alpha=0.4, common_norm=False,
            palette={0:"#E76F51", 1:"#2A9D8F"}, ax=ax2)
ax2.set_title("Age Distribution by Survival", fontweight="bold")

# 3. Survival rate by sex
ax3 = fig.add_subplot(gs[0, 2])
survival_sex = titanic.groupby("sex")["survived"].mean()
bars = ax3.bar(survival_sex.index, survival_sex.values,
               color=["#457B9D","#E63946"], edgecolor="white", width=0.5)
ax3.bar_label(bars, fmt="%.1%", padding=3, fontweight="bold")
ax3.set_title("Survival Rate by Sex", fontweight="bold")
ax3.set_ylabel("Survival Rate"); ax3.set_ylim(0, 1)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

# 4. Fare violin by class
ax4 = fig.add_subplot(gs[1, 0])
sns.violinplot(data=titanic, x="pclass", y="fare",
               hue="survived", split=True,
               palette={0:"#E76F51", 1:"#2A9D8F"},
               inner="quart", cut=0, ax=ax4)
ax4.set_yscale("log"); ax4.set_title("Fare Distribution by Class", fontweight="bold")

# 5. Correlation heatmap
ax5 = fig.add_subplot(gs[1, 1])
corr = titanic[["survived","pclass","age","sibsp","parch","fare"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, ax=ax5,
            cbar_kws={"shrink":0.8})
ax5.set_title("Correlation Matrix", fontweight="bold")

# 6. Survival by embarkation
ax6 = fig.add_subplot(gs[1, 2])
sns.barplot(data=titanic, x="embark_town", y="survived",
            hue="pclass", palette="Set2",
            estimator="mean", errorbar="ci", capsize=0.1, ax=ax6)
ax6.set_title("Survival Rate by Embarkation & Class", fontweight="bold")
ax6.set_ylabel("Survival Rate"); ax6.set_xlabel("Embarkation Town")
ax6.tick_params(axis="x", rotation=15)

# 7. Age vs Fare scatter
ax7 = fig.add_subplot(gs[2, 0:2])
scatter = ax7.scatter(
    titanic["age"].fillna(titanic["age"].median()),
    titanic["fare"],
    c=titanic["survived"].map({0:"#E76F51", 1:"#2A9D8F"}),
    s=titanic["pclass"].map({1:80, 2:40, 3:20}),
    alpha=0.5, edgecolors="white", linewidth=0.5
)
s_patch = mpatches.Patch(color="#2A9D8F", label="Survived")
ns_patch = mpatches.Patch(color="#E76F51", label="Not Survived")
ax7.legend(handles=[s_patch, ns_patch], loc="upper right")
ax7.set_yscale("log"); ax7.set_xlabel("Age"); ax7.set_ylabel("Fare (log)")
ax7.set_title("Age vs Fare (size = class, color = survival)", fontweight="bold")

# 8. Logistic regression
ax8 = fig.add_subplot(gs[2, 2])
sns.regplot(data=titanic.dropna(subset=["age"]),
            x="age", y="survived", logistic=True,
            scatter_kws={"alpha":0.15, "s":15},
            line_kws={"color":"#264653", "linewidth":2.5},
            ci=95, ax=ax8)
ax8.set_title("Age → Survival (Logistic)", fontweight="bold")
ax8.set_ylabel("P(Survived)")

plt.savefig("titanic_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
```

---

## Quick Reference Card

```
Setup           plt.subplots(figsize, dpi)   sns.set_theme(style, context, palette)

Matplotlib      ax.plot(x, y, color, lw, ls, marker, alpha, label)
plots           ax.bar / ax.barh   ax.hist(bins, density, cumulative)
                ax.scatter(c, s, alpha, cmap)   ax.boxplot(patch_artist, notch)
                ax.errorbar(yerr, capsize, fmt)   ax.fill_between
                ax.imshow(cmap, vmin, vmax)   ax.pie(autopct, explode)

Axes            set_title / set_xlabel / set_ylabel
                set_xlim / set_ylim   set_xticks / set_xticklabels
                tick_params   spines[].set_visible   grid
                legend(loc, ncol, bbox_to_anchor)
                text / annotate / axhline / axvline / axhspan

Seaborn         sns.scatterplot(hue, style, size, sizes, palette)
relational      sns.lineplot(estimator, errorbar, err_style)

Seaborn         sns.histplot(kde, stat, bins, common_norm)
distribution    sns.kdeplot(fill, bw_adjust, cut, levels)
                sns.ecdfplot(complementary)   sns.rugplot

Seaborn         sns.boxplot(notch, flierprops)
categorical     sns.violinplot(split, inner, scale, cut)
                sns.boxenplot   sns.stripplot(dodge, jitter)
                sns.swarmplot   sns.barplot(estimator, errorbar, capsize)
                sns.countplot(stat)   sns.pointplot(dodge)

Seaborn         sns.heatmap(annot, fmt, cmap, mask, linewidths, center)
matrix          sns.clustermap(method, metric, z_score)

Figure-level    sns.relplot(col, row, col_wrap, kind, height, aspect)
                sns.displot(kind='hist/kde/ecdf')
                sns.catplot(kind='bar/box/violin/strip/swarm')
                sns.pairplot(hue, diag_kind, corner)
                sns.jointplot(kind, marginal_kws)
                sns.FacetGrid → .map()

Regression      sns.regplot(logistic, order, robust, lowess, ci)
                sns.lmplot(col, row, hue)   sns.residplot

Style           plt.style.use('seaborn-v0_8/ggplot/fivethirtyeight/dark_background')
                cmap: viridis plasma RdYlGn coolwarm tab10 Set2
                colors: hex '#264653'  RGB (0.2,0.5,0.8)  named 'red'

Save            fig.savefig('file.png', dpi=300, bbox_inches='tight', facecolor='white')
```

---

*Covers Matplotlib 3.7+ and Seaborn 0.13+. Docs: https://matplotlib.org | https://seaborn.pydata.org*
