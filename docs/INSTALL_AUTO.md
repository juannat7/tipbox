# Installing AUTO-07P

AUTO-07P is required for bifurcation-induced detection in tipbox (fold, hopf, transcritical). Tipbox expects a **source installation** and the `AUTO_DIR` environment variable pointing to the installation root (so it can use the `python` subdir and `import auto`).

**Note:** `pip install git+https://github.com/auto-07p/auto-07p.git` installs the package into your Python environment but does **not** give you the full source tree (no `auto-07p` folder to `cd` into, no `cmds/auto.env.sh`). The steps below use a clone and build instead.

## 1. Clone and build

```bash
git clone https://github.com/auto-07p/auto-07p.git
cd auto-07p
./configure
make
```

## 2. Set AUTO_DIR

Set the environment variable to the **root** of the clone (the directory that contains `python/`, `cmds/`, etc.). For example, if you cloned into `$HOME/auto-07p`:

```bash
export AUTO_DIR=$HOME/auto-07p
```

Add that line to your `~/.bashrc` (or equivalent) so it is set in every shell.

## 3. (Optional) Activate AUTO commands in the shell

To run the `auto` command and use the compiled binaries in your PATH:

```bash
source $AUTO_DIR/cmds/auto.env.sh
```

## 4. Verify

From a Python process that has `AUTO_DIR` set, tipbox will add `$AUTO_DIR/python` to `sys.path` and `import auto`. You can also verify the command-line interface:

```bash
source $AUTO_DIR/cmds/auto.env.sh
auto
```

If you get the AUTO prompt, the installation is correctly configured.



