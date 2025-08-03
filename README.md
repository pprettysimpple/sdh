Implementation requires avx2 support + I've hardcoded keycodes for my keyboard, because it is dull to handle them well.

To build:
`./build.sh && ./a.out`

Or simply compile `main.c` and link it with `X11`. You must supply `-mavx2`, there is no way around it.

There are movements (WASD) + Zoom into center (R/F). It looks like a mess if you zoom in too close.

<img width="1192" height="912" alt="image" src="https://github.com/user-attachments/assets/df016d3e-4d29-49c0-ab9a-c10ca750ef5f" />
