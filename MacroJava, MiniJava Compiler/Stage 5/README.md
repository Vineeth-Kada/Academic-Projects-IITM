## Stage 5: miniRA to MIPS Assembly

This is the final stage of the project. In this stage, we start with programs in `miniRA` format and translate them to `MIPS Assembly`.

To generate MIPS Assembly code for `A.miniRA` file, run the following commands:

```bash
javac A5.java

java A5 < A.miniRA > A.s
```

To ensure that the generated `.s` program is semantically equivalent to the `miniRA` program, one can use the MIPS interpreter [(download link)](https://pages.cs.wisc.edu/~larus/spim.html) to compare the output of `A.miniRA` with the output of the `.s` program generated by the compiler.