import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Floating-point networks: exact values, surprising gradients

    ![Claim-by-claim result](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-g89qqA6qmD-floating-point-networks-with-automatic-differentiation-can-represent-almost/main/reports/floating-point-networks/images/headline.svg)

    This tutorial explains the central idea and opens with the formal
    evidence. It is self-contained: the displayed results are embedded
    from the completed Hugging Face CPU runs, so opening the notebook does
    **not** rerun the expensive reproduction.

    The paper asks whether a floating-point network can return a chosen
    value \(f^*(x)\) while automatic differentiation reports an
    independently chosen gradient \(g^*(x)\). Exact real arithmetic makes
    those quantities tightly related. Intermediate rounding makes the
    executed forward and backward programs different enough to separate
    them.
    """)
    return


@app.cell
def _():
    formal_rows = [
        {
            "x": 2.0**-20,
            "target value": 0.25,
            "observed value": 0.25,
            "upstream": 0.5,
            "target gradient": 1.0,
            "observed gradient": 1.0,
        },
        {
            "x": 0.25,
            "target value": -0.5,
            "observed value": -0.5,
            "upstream": 1.0,
            "target gradient": -2.0,
            "observed gradient": -2.0,
        },
        {
            "x": 0.5,
            "target value": 1.0,
            "observed value": 1.0,
            "upstream": 2.0,
            "target gradient": 0.5,
            "observed gradient": 0.5,
        },
        {
            "x": 1.0,
            "target value": -2.0,
            "observed value": -2.0,
            "upstream": 4.0,
            "target gradient": -1.0,
            "observed gradient": -1.0,
        },
        {
            "x": 2.0,
            "target value": 4.0,
            "observed value": 4.0,
            "upstream": 0.25,
            "target gradient": 4.0,
            "observed gradient": 4.0,
        },
        {
            "x": 8.0,
            "target value": -8.0,
            "observed value": -8.0,
            "upstream": 8.0,
            "target gradient": -0.25,
            "observed gradient": -0.25,
        },
    ]
    return (formal_rows,)


@app.cell
def _(formal_rows, mo):
    mo.vstack(
        [
            mo.md(
                r"""
                ## The 13-layer result

                Exact indicators select one branch for each input. Value
                branches return the target with zero AD gradient; calibrated
                gradient branches return zero with the target AD gradient. A
                final affine layer joins them. Every number below is an exact
                float32 match, not a tolerance-based comparison.
                """
            ),
            mo.ui.table(formal_rows, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    point = mo.ui.slider(
        start=0,
        stop=5,
        step=1,
        value=0,
        label="Inspect one formal test point",
        show_value=True,
    )
    point
    return (point,)


@app.cell
def _(formal_rows, mo, point):
    row = formal_rows[point.value]
    mo.callout(
        mo.md(
            f"""
            At **x = {row["x"]}**, the network returns
            **{row["observed value"]}** and AD reports
            **{row["observed gradient"]}**, exactly matching the independent
            targets. The upstream loss derivative is **{row["upstream"]}**.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why arithmetic order matters

    The independent interpreter executes each float32 multiply and add in
    explicit left-to-right order in both forward and backward passes.
    That is part of the mathematical object being tested. An optimized
    matrix kernel may associate the same operations differently.

    Claim 6 isolates this mechanism in binary64. For the deterministic
    operands

    \[
    a=-3.390181137426782\times10^{-49},\quad
    b=1.9001924576251575\times10^{-36},\quad
    c=2.9840622466904965\times10^{-46},
    \]

    `(a*b)*c` is `-1.9223318928897442e-130`, while `a*(b*c)` is
    `-1.9223318928897446e-130`. Reconstructing the inputs as exact rational
    numbers makes the products equal, proving that intermediate rounding
    causes the difference. Powers of two `(2, 0.5, 8)` are the
    non-triggering control.
    """)
    return


@app.cell
def _(mo):
    activation_rows = [
        {"activation": "ReLU", "Condition 2": "4/4", "Condition 3": "first ✓", "autograd": "agree"},
        {"activation": "ELU", "Condition 2": "4/4", "Condition 3": "second ✓", "autograd": "agree"},
        {"activation": "GELU", "Condition 2": "4/4", "Condition 3": "second ✓", "autograd": "agree"},
        {"activation": "Swish", "Condition 2": "4/4", "Condition 3": "second ✓", "autograd": "agree"},
        {"activation": "Sigmoid", "Condition 2": "4/4", "Condition 3": "second ✓", "autograd": "agree"},
        {"activation": "tanh", "Condition 2": "4/4", "Condition 3": "second ✓", "autograd": "agree"},
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## What generalized—and what did not

                All six named activations pass the directly tested numerical
                witnesses. The constant activation control is rejected.
                """
            ),
            mo.ui.table(activation_rows, selection=None),
            mo.callout(
                "Claims 1–5 remain BLOCKED because finite checks cannot prove "
                "the universal target-map and complete-domain quantifiers. "
                "Claim 6 is VERIFIED.",
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Reproduce or inspect

    The formal fixed command is:

    ```bash
    uv sync --frozen --no-dev && uv run --frozen --no-sync python run_reproduction.py
    ```

    The [illustrated report](https://github.com/MachineLearning-Nerd/icml26-repro-g89qqA6qmD-floating-point-networks-with-automatic-differentiation-can-represent-almost/blob/main/reports/floating-point-networks/report.md)
    documents the architecture, controls, source hashes, compute, and
    theorem-level limitations. The evaluator-visible standard-library
    verifier is `space_candidate/code/verify_current.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
