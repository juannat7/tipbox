# Installing AUTO-07P

AUTO-07P is required for proper bifurcation-induced detection (distinguishing fold, hopf, and transcritical).

1. Install package
```
pip install git+https://github.com/auto-07p/auto-07p.git
```

2. Configure, compile, and make the software
```
cd auto-07p
./configure
make
```

3. Activate AUTO installation
```
source [path-to-auto-07p]/cmds/auto.env.sh
```

4. Verify installation. If it works, you will end up in the AUTO Python prompt. It means you have AUTO properly configured.
```
auto
```



