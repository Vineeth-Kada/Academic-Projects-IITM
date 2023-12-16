## Stage 1: MacroJava to MiniJava
MacroJava is a subset of Java extended with C style macros. The meaning of a MacroJava program is given by its meaning as a Java program (after macro processing). Overloading is not allowed in MacroJava. The MacroJava statement System.out.println( ... ); can only print integers. The MacroJava expression e1 & e2 is of type boolean, and both e1 and e2 must be of type boolean. MacroJava supports both inline as well as C style comments, but does not support nested comments.

The goal of stage 1 is to write a MacroJava to MiniJava translator using Flex and Bison. 

Commands to convert the MacroJava file (say X.java) to MiniJava (say Y.java)
```bash
$ bison -d A1.y

$ flex A1.l

$ gcc A1.tab.c lex.yy.c -lfl -o A1

$ ./A1 < X.java > Y.java 
```