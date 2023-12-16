%{
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<stdarg.h>
#include<assert.h>

int yylex();
int yyerror();

// Takes a set of strings (char *) and merge them by adding a space in between them
char* merge(int count, ...)
{
    va_list valist;
	
	// Find the Space Required by the Merged String
	int totalLen = count;	// For Spaces and Null character
    va_start(valist, count);
    for(int i = 0; i < count; i++) totalLen += strlen(va_arg(valist, char*));
    va_end(valist);
	
	// Allocate Required Space
	char * merged = (char *) malloc(totalLen);
	merged[0] = '\0';
	
	// Merge Strings one by one
	va_start(valist, count);
    for(int i = 0; i < count; i++){
		strcat(merged, va_arg(valist, char*));
		if(i != count-1) strcat(merged, " ");
	}
    va_end(valist);

    return merged;
}

// Linked List of strings (char *)
typedef struct List{
	char* val;
	struct List* next;
}List;

void Append(List* L, char* newNode){
	List* ptr = L;
	while(ptr->next != NULL) ptr = ptr->next;
	ptr->next = (List*) malloc(sizeof(List));
	ptr->next->val = newNode;
	ptr->next->next = NULL;
}
List* create(){
	List* ptr = (List *) malloc(sizeof(List));
	ptr->val = NULL; ptr->next = NULL;
	return ptr;
}

// Number of tokens in comma separated values
int CountTokens(char* str){
	if(str == NULL) return 0;
	int n=strlen(str);
	if(n == 0) return 0;
	int ans = 1;
	for(int i=0; i<n; i++) if(str[i] == ',') ans++;
	return ans;
}

//	Note: Once you use tokenizer string will literally be tokenized you can no longer access the original string
char** Tokenize(char* str, int cnt){
	if(cnt == 0) return NULL;
	char** List = (char **) malloc(sizeof(char *) * cnt);
	char* token = strtok(str, " , ");
	for(int i=0; i < cnt; i++){
		List[i] = token;
		token = strtok(NULL, " , ");
	}
	return List;
}

// Convert a space(" ") separated string into a linked list of tokens with a DUMMY at the beginning
List* TokenizeToList(char* str){
	List* head = (List*) malloc(sizeof(List));
	List* tail = head;
	tail->next = NULL;
	if(str == NULL) return head;
	char* token = strtok(str, " ");
	while(token != NULL){
		tail->next = (List*) malloc(sizeof(List));
		tail = tail->next;
		tail->val = token;
		tail->next = NULL;
		token = strtok(NULL, " ");
	}
	return head;
}

typedef struct Macro{
    char* Name;
    int Argc;
   	char** ArgList;
	List* Expansion;
}Macro;

Macro MacroList[100000];	// Macro List
int MacroCount = 0;	// Macro Count

// Add a Macro to the Macro List
void addToML(char* __Name, int __Argc, char* __ArgList, char* __Expansion){
	MacroList[MacroCount].Name = __Name;
	MacroList[MacroCount].Argc = __Argc;
	MacroList[MacroCount].ArgList = Tokenize(__ArgList, __Argc);
	MacroList[MacroCount].Expansion = TokenizeToList(__Expansion);
	MacroCount++;
}

%}

%union {
	char *text;
	struct List* LinkedList;
}

%token <text> CLASS LCB RCB PUBLIC STATIC VOID MAIN LP RP STRING LSB RSB SYS_OUT_PRINT SEMICOLON EXTENDS COMMA RETURN INT BOOL EQ IF ELSE WHILE BWAND BWOR NEQ LEQ ADD SUB MUL DIV DOT LENGTH TRUE FALSE THIS NEW NOT DEFS DEFS0 DEFS1 DEFS2 INTEGER IDENTIFIER COMMENT DEFE DEFE0 DEFE1 DEFE2

%type<text>  MacroDefExpression IDENTIFIER_3_PLUS MacroDefStatement MacroDefinition PrimaryExpression Expression Statement_Asterate Statement Type Goal MacroDefinition_Asterate TypeDeclaration_Asterate MainClass TypeDeclaration MethodDeclaration_Asterate MethodDeclaration Type_Identifier_SC_Asterate Type_Identifier_Series	Type_Identifier_Series_NonEmpty MethodBody MacroCall

%type<LinkedList> Expression_Series Expression_Series_NonEmpty

%%

Goal:  MacroDefinition_Asterate MainClass TypeDeclaration_Asterate
{
	$$ = merge(3, $1, $2, $3);
	printf("%s\n", $$);	// Goal is the final program. So we have to print at this point
};

MacroDefinition_Asterate:	{/*Epsilon Production*/ $$ = merge(1, ""); }
						|	MacroDefinition_Asterate MacroDefinition	{ $$ = merge(2, $1, $2); };
							
TypeDeclaration_Asterate:	{ /*Epsilon Production*/ $$ = merge(1, ""); }
						|	TypeDeclaration_Asterate TypeDeclaration	{ $$ = merge(2, $1, $2); };
							
MainClass:	CLASS IDENTIFIER LCB PUBLIC STATIC VOID MAIN LP STRING LSB RSB IDENTIFIER RP LCB SYS_OUT_PRINT LP Expression RP SEMICOLON RCB RCB { $$ = merge(21, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21); };

TypeDeclaration:	CLASS IDENTIFIER LCB Type_Identifier_SC_Asterate MethodDeclaration_Asterate RCB						{ $$ = merge(6, $1, $2, $3, $4, $5, $6); }
            	|	CLASS IDENTIFIER EXTENDS IDENTIFIER LCB Type_Identifier_SC_Asterate MethodDeclaration_Asterate RCB	{ $$ = merge(8, $1, $2, $3, $4, $5, $6, $7, $8); };

MethodDeclaration_Asterate:	{ /*Epsilon Production*/ $$ = merge(1, ""); }
						|	MethodDeclaration_Asterate	MethodDeclaration	{ $$ = merge(2, $1, $2); };

MethodDeclaration:	PUBLIC Type IDENTIFIER LP Type_Identifier_Series RP LCB MethodBody RETURN Expression SEMICOLON RCB	{ $$ = merge(12, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12); }
				|	PUBLIC Type IDENTIFIER LP Type_Identifier_Series RP LCB RETURN Expression SEMICOLON RCB 			{ $$ = merge(11, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11); };

MethodBody:	Type IDENTIFIER SEMICOLON MethodBody	{ $$ = merge(4, $1, $2, $3, $4); }
		|	MethodBody Statement					{ $$ = merge(2, $1, $2); }
		|	Type IDENTIFIER SEMICOLON				{ $$ = merge(3, $1, $2, $3); }
		|	Statement								{ $$ = merge(1, $1); };

Type_Identifier_SC_Asterate:	{ /*Epsilon Production*/ $$ = merge(1, ""); }
							|	Type_Identifier_SC_Asterate Type IDENTIFIER SEMICOLON	{ $$ = merge(4, $1, $2, $3, $4); };

Type:	INT LSB RSB	{ $$ = merge(3, $1, $2, $3); }
	|	BOOL		{ $$ = merge(1, $1); }
	|	INT			{ $$ = merge(1, $1); }
	|	IDENTIFIER	{ $$ = merge(1, $1); };

Type_Identifier_Series:	{ /*Epsilon Production*/ $$ = merge(1, ""); }
                	|	Type_Identifier_Series_NonEmpty	{ $$ = merge(1, $1); };
					
Type_Identifier_Series_NonEmpty:	Type IDENTIFIER	{ $$ = merge(2, $1, $2); }
                        		|	Type_Identifier_Series_NonEmpty COMMA Type IDENTIFIER	{ $$ = merge(4, $1, $2, $3, $4); };
								
Statement:	LCB Statement_Asterate RCB								{ $$ = merge(3, $1, $2, $3); }
		|	SYS_OUT_PRINT LP Expression RP SEMICOLON				{ $$ = merge(5, $1, $2, $3, $4, $5); }
		|	IDENTIFIER EQ Expression SEMICOLON						{ $$ = merge(4, $1, $2, $3, $4); }
		|	IDENTIFIER LSB Expression RSB EQ Expression SEMICOLON	{ $$ = merge(7, $1, $2, $3, $4, $5, $6, $7); }
		|	IF LP Expression RP Statement							{ $$ = merge(5, $1, $2, $3, $4, $5); }
		|	IF LP Expression RP Statement ELSE Statement			{ $$ = merge(7, $1, $2, $3, $4, $5, $6, $7); }
		|	WHILE LP Expression RP Statement						{ $$ = merge(5, $1, $2, $3, $4, $5); }
		|	MacroCall  SEMICOLON									{ $$ = merge(2, $1, $2); };

Statement_Asterate:	{ /*Epsilon Production*/ $$ = merge(1, ""); }
				|	Statement_Asterate Statement	{ $$ = merge(2, $1, $2); };

Expression_Series:	{	$$ = create();	}
                |	Expression_Series_NonEmpty	{ $$ = $1; };

Expression_Series_NonEmpty:	Expression	{ $$ = create(); Append($$, $1); }
                        |	Expression_Series_NonEmpty COMMA Expression	{ $$ = $1; Append($$, $3); };

Expression:	PrimaryExpression BWAND PrimaryExpression	{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression BWOR PrimaryExpression	{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression NEQ PrimaryExpression		{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression LEQ PrimaryExpression		{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression ADD PrimaryExpression		{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression SUB PrimaryExpression		{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression MUL PrimaryExpression		{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression DIV PrimaryExpression		{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression LSB PrimaryExpression RSB	{ $$ = merge(4, $1, $2, $3, $4); }
		|	PrimaryExpression DOT LENGTH				{ $$ = merge(3, $1, $2, $3); }
		|	PrimaryExpression							{ $$ = merge(1, $1); }
		|	PrimaryExpression DOT IDENTIFIER LP Expression_Series RP	{
				/* Expression_Series is Linked List of String */
				$$ = merge(4, $1, $2, $3, $4);
				for(List* ptr = $5->next; ptr != NULL; ptr = ptr->next){
					$$ = merge(2, $$, ptr->val);
					if(ptr->next != NULL) $$ = merge(2, $$, ",");
				}
				$$ = merge(2, $$, $6);
			}
		|	MacroCall	{$$ = merge(1, $1);};

MacroCall:	IDENTIFIER LP Expression_Series RP{
				$$ = merge(1, "");
				int yes = 0;	// = 1, If the give identifier Matches a Macro Definition
				for(int I=0; I<MacroCount; I++){	// Go through all entries in macro list and match macro name and the current identifier
					if(!strcmp(MacroList[I].Name, $1)){
						// Finding Size of Expression Series
						int NoOfExpr = 0; List* ptr; for(ptr = $3->next; ptr != NULL; ptr = ptr->next) NoOfExpr++;
						
						// Storing the series in the form of an array
						char** Expression_Array = (char **) malloc(sizeof(char*) * NoOfExpr);
						ptr = $3->next;
						for(int i=0; i<NoOfExpr; i++, ptr = ptr->next) Expression_Array[i] = ptr->val;
						
						for(ptr = MacroList[I].Expansion->next; ptr != NULL; ptr = ptr->next){
							int isArg = 0;
							for(int i=0; i < MacroList[I].Argc; i++){
								if(! strcmp(MacroList[I].ArgList[i], ptr->val)){
									$$ = merge(2, $$, Expression_Array[i]);
									isArg = 1;
									break;
								}
							}
							if(! isArg) $$ = merge(2, $$, ptr->val);
						}
						
						yes=1;
						break;
					}
				}
				// If it is not a macro then it may a function call so keep it as it is
				if(! yes){
					$$ = merge(2, $1, $2);
					for(List* ptr = $3->next; ptr != NULL; ptr = ptr->next){
						$$ = merge(2, $$, ptr->val);
						if(ptr->next != NULL) $$ = merge(2, $$, ",");
					}
					$$ = merge(2, $$, $4);
				}
			};

PrimaryExpression:	INTEGER							{ $$ = merge(1, $1); }
				|	TRUE							{ $$ = merge(1, $1); }
				|	FALSE							{ $$ = merge(1, $1); }
				|	IDENTIFIER						{ $$ = merge(1, $1); }		
				|	THIS							{ $$ = merge(1, $1); }
				|	NEW INT LSB Expression RSB		{ $$ = merge(5, $1, $2, $3, $4, $5); }
				|	NEW IDENTIFIER LP RP			{ $$ = merge(4, $1, $2, $3, $4); }
				|	NOT Expression					{ $$ = merge(2, $1, $2); }
				|	LP Expression RP				{ $$ = merge(3, $1, $2, $3); };

MacroDefinition:	MacroDefExpression	{ /*We have to exclude MacroDefinition so take empty string*/ $$ = merge(1, "");}
				|	MacroDefStatement	{$$ = merge(1, "");};

MacroDefStatement:	DEFS IDENTIFIER LP IDENTIFIER_3_PLUS RP LCB Statement_Asterate RCB 				{ addToML($2, CountTokens($4), $4, merge(3, $6, $7, $8)); }
				|	DEFS0 IDENTIFIER LP RP LCB Statement_Asterate RCB								{ addToML($2, 0, NULL, merge(3, $5, $6, $7)); }
				|	DEFS1 IDENTIFIER LP IDENTIFIER RP LCB Statement_Asterate RCB 					{ addToML($2, 1, $4, merge(3, $6, $7, $8));}
				|	DEFS2 IDENTIFIER LP IDENTIFIER COMMA IDENTIFIER RP LCB Statement_Asterate RCB 	{ addToML($2, 2, merge(3, $4, $5, $6), merge(3, $8, $9, $10)); };

MacroDefExpression:	DEFE IDENTIFIER LP IDENTIFIER_3_PLUS RP LP Expression RP 						{ addToML($2, CountTokens($4), $4, merge(3, $6, $7, $8)); }
				|	DEFE0 IDENTIFIER LP RP LP Expression RP											{ addToML($2, 0, NULL, merge(3, $5, $6, $7)); }
				|	DEFE1 IDENTIFIER LP IDENTIFIER RP LP Expression RP 								{ addToML($2, 1, $4, merge(3, $6, $7, $8));}
				|	DEFE2 IDENTIFIER LP IDENTIFIER COMMA IDENTIFIER RP LP Expression RP 			{ addToML($2, 2, merge(3, $4, $5, $6), merge(3, $8, $9, $10)); };
					
IDENTIFIER_3_PLUS:	IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER	{ $$ = merge(5, $1, $2, $3, $4, $5); }
				|	IDENTIFIER_3_PLUS	COMMA IDENTIFIER			{ $$ = merge(3, $1, $2, $3); };
%%

int yyerror(char *s)
{
	printf("//Failed to parse input code");
	return 0;
}

int main(int argc, char **argv)
{
	yyparse();
	return 0;
}
