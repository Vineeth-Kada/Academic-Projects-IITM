%{
#include "A1.tab.h"
#include<string.h>

void init(){
    yylval.text = (char *) malloc(sizeof(char) * ( strlen(yytext)+2 ));
    strcpy(yylval.text, yytext);
}
%}

%%

"class" { init(); return CLASS ;}
"{" { init(); return LCB ;}
"}" { init(); return RCB ;}
"public" { init(); return PUBLIC ;}
"static" { init(); return STATIC ;}
"void" { init(); return VOID ;}
"main" { init(); return MAIN ;}
"(" { init(); return LP ;}
")" { init(); return RP ;}
"String" { init(); return STRING ;}
"[" { init(); return LSB ;}
"]" { init(); return RSB ;}
"System.out.println" { init(); return SYS_OUT_PRINT ;}
";" { init(); return SEMICOLON ;}
"extends" { init(); return EXTENDS ;}
"," { init(); return COMMA ;}
"return" { init(); return RETURN;}
"int" { init(); return INT ;}
"boolean" { init(); return BOOL ;}
"=" { init(); return EQ ;}
"if" { init(); return IF ;}
"else" { init(); return ELSE ;}
"while" { init(); return WHILE ;}
"&&" { init(); return BWAND ;}
"||" { init(); return BWOR ;}
"!=" { init(); return NEQ ;}
"<=" { init(); return LEQ ;}
"+" { init(); return ADD ;}
"-" { init(); return SUB ;}
"*" { init(); return MUL ;}
"/" { init(); return DIV ;}
"." { init(); return DOT ;}
"length" { init(); return LENGTH ;}
"true" { init(); return TRUE ;}
"false" { init(); return FALSE ;}
"this" { init(); return THIS ;}
"new" { init(); return NEW ;}
"!" { init(); return NOT ;}
"#defineStmt0" { init(); return DEFS0 ;}
"#defineStmt1" { init(); return DEFS1 ;}
"#defineStmt2" { init(); return DEFS2 ;}
"#defineStmt" { init(); return DEFS ;}
"#defineExpr0" { init(); return DEFE0 ;}
"#defineExpr1" { init(); return DEFE1 ;}
"#defineExpr2" { init(); return DEFE2 ;}
"#defineExpr" { init(); return DEFE ;}
0|[1-9][0-9]* { init(); return INTEGER ;}
[a-zA-Z][a-zA-Z0-9_]* { init(); return IDENTIFIER ;}
"//".* { }
[ \t\n] { }

%%