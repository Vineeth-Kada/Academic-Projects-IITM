Five stages of compilation:
- MacroJava to MiniJava
- Type Checker
- MiniJava to microIR
- microIR to miniRA
- miniRA to MIPS assembly

See the `Language Specifications` folder for the specifications of MacroJava, MiniJava, microIR and miniRA. See the `Test Programs` folder for sample programs in MacroJava, MiniJava, microIR and miniRA.

Requirements: `javacc, javac`

To run the compiler one needs to run each stage separately and pass the output of the previous stage as input to the next stage. The commands to run each stage are provided in the respective folders.

// Aside

AST visitors are modified to build the compiler for MiniJava (stages 2 to 5). Generating AST visitors for MiniJava (/ microIR / miniRA / MIPS) using JavaCC and JTB:

1. **Define Your Language Grammar in `minijava.jj`**:
   - You begin by defining the grammar of your target language in a `.jj` file. This file specifies how your language's syntax is structured.

2. **Generate JavaCC's input grammar, visitors and JTB Output File Using JTB**:
   - JTB takes the parser specifications written for JavaCC and enhances them with additional capabilities for building and working with syntax trees.
   - Running the following command generates the JavaCC's input grammar, AST classes (`syntaxtree/`) and visitor (`visitor/`) interfaces:
     ```
     java -jar jtb132.jar minijava.jj
     ```
   - [Link to download jtb132](http://www.java2s.com/example/jar/j/download-jtb132jar-file.html)

3. **Generate the Lexer and Parser Using JavaCC**:
   - Use JavaCC to generate the lexer and parser components based on the generated `jtb.out.jj` file.
   - JavaCC generates LL parsers, specifically LL(k) parsers, where 'k' represents the number of lookahead tokens used to make parsing decisions.
   - Run the following command to generate the lexer and parser:
     ```
     javacc jtb.out.jj
     ```
   - This step creates a set of Java files, including `MiniJavaParser.java`, `MiniJavaParserConstants.java`, and `MiniJavaParserTokenManager.java`, which are used for parsing your language.