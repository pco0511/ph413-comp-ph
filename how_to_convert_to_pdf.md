
- direct
```bash
jupyter nbconvert --to pdf {ipynb_file_name}
```

- convert to .tex and complile
```bash
jupyter nbconvert --to latex {ipynb_file_name}
xelatex {tex_file_name}
```