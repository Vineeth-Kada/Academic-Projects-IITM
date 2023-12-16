## Stage 2: Type Checking

In stage 2, the implementation of a type checker for the MiniJava compiler will be undertaken.

To type check 'A.java' file, run the following commands:

```bash
javac A2.java

java A2 < A.java
```

JavaCC (Java Compiler Compiler) and JTB (Java Tree Builder) are employed for generating the parser and visitors for MiniJava based on the grammar outlined in the `minjava.jj` file. Subsequently, the visitor pattern is utilized to traverse the AST and execute type checking. `visitor/GJDepthFirst1.java` and `visitor/GJDepthFirst2.java` are the two visitors that are implemented for this purpose. Specifically in those files, `// User-generated visitor methods below` is the section where the type checking code is written in the generated visitors. These two visitors are called by `A2.java`. We need two visitors because, functions and classes can be declared after calling them in MiniJava. Hence, we need to traverse the AST twice to check for type errors. First traversal is for finding the function declarations and second traversal is for checking the type errors.
