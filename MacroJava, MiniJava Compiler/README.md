Five stages of compilation:
- MacroJava to MiniJava
- Type Checker
- MiniJava to microIR
- microIR to miniRA
- miniRA to MIPS assembly

See the [Language Specifications](https://github.com/Vineeth-Kada/MacroJava-Compiler/tree/main/Language%20Specifications) folder for the specifications of MacroJava, MiniJava, microIR and miniRA.

## Part 1: MacroJava to MiniJava
Translate using flex and bison

Commands to convert the MacroJava file (say X.java) to MiniJava (say Y.java)
```bash
# Go to Part 1 Directory

$ bison -d A1.y

$ flex A1.l

$ gcc A1.tab.c lex.yy.c -lfl -o A1

$ ./A1 < X.java > Y.java 
```