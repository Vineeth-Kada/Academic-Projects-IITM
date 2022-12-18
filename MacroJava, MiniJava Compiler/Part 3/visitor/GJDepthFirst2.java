package visitor;
import syntaxtree.*;
import java.util.*;
import java.util.function.BiFunction;

/**
 * Provides default methods which visit each node in the tree in depth-first
 * order.  Your visitors may extend this class.
 */
public class GJDepthFirst2<R,A> implements GJVisitor<R,A> {
   
   //
   // Auto class visitors--probably don't need to be overridden.
   //
   public R visit(NodeList n, A argu) {
      R _ret=null;
      int _count=0;
      for ( Enumeration<Node> e = n.elements(); e.hasMoreElements(); ) {
         e.nextElement().accept(this,argu);
         _count++;
      }
      return _ret;
   }

   public R visit(NodeListOptional n, A argu) {
      if ( n.present() ) {
         R _ret=null;
         int _count=0;
         for ( Enumeration<Node> e = n.elements(); e.hasMoreElements(); ) {
            e.nextElement().accept(this,argu);
            _count++;
         }
         return _ret;
      }
      else
         return null;
   }

   public R visit(NodeOptional n, A argu) {
      if ( n.present() )
         return n.node.accept(this,argu);
      else
         return null;
   }

   public R visit(NodeSequence n, A argu) {
      R _ret=null;
      int _count=0;
      for ( Enumeration<Node> e = n.elements(); e.hasMoreElements(); ) {
         e.nextElement().accept(this,argu);
         _count++;
      }
      return _ret;
   }

   public R visit(NodeToken n, A argu) { return (R) n.tokenImage; }
   
   //
	// User-generated visitor methods below
	//

   
	HashMap<String, ArrayList<String>> ownFuns;   // Own Functions - Present in the scope of current class
	HashMap<String, HashMap<String, Integer>> GV; // Global variables of current class
	HashMap<String, String> parent;  // parent of current class - default null
   HashMap<String, Integer> FunTable;

   public GJDepthFirst2(
		HashMap<String, ArrayList<String>> __ownFuns,
		HashMap<String, HashMap<String, Integer>> __GV,
		HashMap<String, String> __parent,
      HashMap<String, Integer> __FunTable
	){
		ownFuns = __ownFuns; GV = __GV; parent = __parent; FunTable = __FunTable;
	}

   // Get offset of a variable name in a given class. We can retreive it from GV Table by traversing till the super ancestors
   public int getOffset(String className, String varName){
      while(true){
         if(GV.get(className).containsKey(varName)){
            return 4 * (countAllGV(parent.get(className)) + GV.get(className).get(varName));
         }
         className = parent.get(className);
      }
   }

   // Fill the fuction table of the current object with function labels of this class and parent classes
   public void fillFunTable(String className, int baseFT){  
      Set<String> vis = new HashSet<String>();
      do{
         for(String fun : ownFuns.get(className)){
            if(! vis.contains(fun)){
               vis.add(fun);
               System.out.println("HSTORE TEMP " + baseFT + " " + FunTable.get(fun)*4 + " " + className + "_" + fun);
            }
         }
         className = parent.get(className);
      }while(className != null);
   }
   
   // Count the global variables in this class and all parent classes
   public int countAllGV(String className){
      if(className == null) return 0;
      int cnt = 0;
      do{
         cnt += GV.get(className).size();
         className = parent.get(className);
      }while(className != null);
      return cnt;
   }
   
	int level; // level - 0 for global vars and 1 for local vars
	String className = null;   // Current Class we are dealing with
   String funName = null;
   HashMap< String, HashMap<String, HashMap<String, Integer>> > LV = new HashMap<>();   // Function Local Var: Class, Function, Var Name - Temp Number
   HashMap< String, HashMap<String, HashMap<String, Integer>> > FP = new HashMap<>();   // Function Formal Param: Class, Function, Arg Name - Temp Number
   int FPcounter = 1;
   int labelCounter = 2;
   int tempCounter = 100;
   
   public void debug(){
      System.out.println("Offsets of LV = " + LV);
      System.out.println("Offsets of FP = " + FP);
      System.out.println("GV = " + GV);
      System.out.println("OwnFuns = " + ownFuns);
   }
	
   /**
	* f0 -> MainClass()
	* f1 -> ( TypeDeclaration() )*
	* f2 -> <EOF>
	*/
	public R visit(Goal n, A argu) {
		R _ret = null;
      System.out.println("MAIN");
		n.f0.accept(this, argu);
      System.out.println("END\n");
		n.f1.accept(this, argu);
		n.f2.accept(this, argu);
		return _ret;
	}

   /**
    * f0 -> "class"
    * f1 -> Identifier()
    * f2 -> "{"
    * f3 -> "public"
    * f4 -> "static"
    * f5 -> "void"
    * f6 -> "main"
    * f7 -> "("
    * f8 -> "String"
    * f9 -> "["
    * f10 -> "]"
    * f11 -> Identifier()
    * f12 -> ")"
    * f13 -> "{"
    * f14 -> PrintStatement()
    * f15 -> "}"
    * f16 -> "}"
    */
	public R visit(MainClass n, A argu) {
		R _ret=null;
		n.f0.accept(this, argu);
		n.f1.accept(this, argu);
		n.f2.accept(this, argu);
		n.f3.accept(this, argu);
		n.f4.accept(this, argu);
		n.f5.accept(this, argu);
		n.f6.accept(this, argu);
		n.f7.accept(this, argu);
		n.f8.accept(this, argu);
		n.f9.accept(this, argu);
		n.f10.accept(this, argu);
		n.f11.accept(this, argu);
		n.f12.accept(this, argu);
		n.f13.accept(this, argu);
		n.f14.accept(this, argu);
		n.f15.accept(this, argu);
		n.f16.accept(this, argu);
		return _ret;
	}

   /** No need to do anything
    * f0 -> ClassDeclaration()
    *       | ClassExtendsDeclaration()
    */
	public R visit(TypeDeclaration n, A argu) {
		R _ret=null;
		n.f0.accept(this, argu);
		return _ret;
	}

   /**
    * f0 -> "class"
    * f1 -> Identifier()
    * f2 -> "{"
    * f3 -> ( VarDeclaration() )*
    * f4 -> ( MethodDeclaration() )*
    * f5 -> "}"
    */
	public R visit(ClassDeclaration n, A argu) {
		R _ret=null;
      level = 0;
		n.f0.accept(this, argu);
		className = (String) n.f1.accept(this, argu);
      LV.put(className, new HashMap<>());
      FP.put(className, new HashMap<>());
		n.f2.accept(this, argu);
		n.f3.accept(this, argu);
		level = 1;
		n.f4.accept(this, argu);
		n.f5.accept(this, argu);
		return _ret;
	}

   /** Hopefully nothing else 
    * f0 -> "class"
    * f1 -> Identifier()
    * f2 -> "extends"
    * f3 -> Identifier()
    * f4 -> "{"
    * f5 -> ( VarDeclaration() )*
    * f6 -> ( MethodDeclaration() )*
    * f7 -> "}"
    */
	public R visit(ClassExtendsDeclaration n, A argu) {
		R _ret=null;
      level = 0;
		n.f0.accept(this, argu);
		className = (String) n.f1.accept(this, argu);
      LV.put(className, new HashMap<>());
      FP.put(className, new HashMap<>());
		n.f2.accept(this, argu);
      n.f3.accept(this, argu);
		n.f4.accept(this, argu);
		n.f5.accept(this, argu);
		level = 1;
		n.f6.accept(this, argu);
		n.f7.accept(this, argu);
		return _ret;
	}

   /** DONE 
    * f0 -> Type()
    * f1 -> Identifier()
    * f2 -> ";"
    */
	public R visit(VarDeclaration n, A argu) {
		R _ret=null;
		String idType = (String) n.f0.accept(this, argu);
		String idName = (String) n.f1.accept(this, argu);
      
      if(level == 1){ // Function Local Variables
         LV.get(className).get(funName).put(idName, tempCounter++);
      }
      
		n.f2.accept(this, argu);
		return _ret;
	}

   /** DONE
    * f0 -> AccessType()
    * f1 -> Type()
    * f2 -> Identifier()
    * f3 -> "("
    * f4 -> ( FormalParameterList() )?
    * f5 -> ")"
    * f6 -> "{"
    * f7 -> ( VarDeclaration() )*
    * f8 -> ( Statement() )*
    * f9 -> "return"
    * f10 -> Expression()
    * f11 -> ";"
    * f12 -> "}"
    */
	public R visit(MethodDeclaration n, A argu) {
		R _ret=null;
		n.f0.accept(this, argu);
		n.f1.accept(this, argu);
      funName = (String) n.f2.accept(this, argu);
      
      LV.get(className).put(funName, new HashMap<>());
      FP.get(className).put(funName, new HashMap<>());
      FPcounter = 1;
      
		n.f3.accept(this, argu);
		n.f4.accept(this, argu); // Formal Param List
      System.out.println(className+"_"+funName+" ["+(FP.get(className).get(funName).size() + 1)+"]");
      System.out.println("BEGIN");
		n.f5.accept(this, argu);
		n.f6.accept(this, argu);
		n.f7.accept(this, argu);
		n.f8.accept(this, argu);
		n.f9.accept(this, argu); // Return
		Integer exprResult = (Integer) n.f10.accept(this, argu); // Return expr
      System.out.println("RETURN TEMP " + exprResult + "\nEND\n");
      // if(! isAncestor(expReturnType, actualReturnType)) ERROR("Method Decl.");
		n.f11.accept(this, argu);
		n.f12.accept(this, argu);
		return _ret;
	}

	/**
	* f0 -> FormalParameter()
	* f1 -> ( FormalParameterRest() )*
	*/
	public R visit(FormalParameterList n, A argu) {
		R _ret=null;
		n.f0.accept(this, argu);
		n.f1.accept(this, argu);
		return _ret;
	}

   /** DONE 
    * f0 -> Type()
    * f1 -> Identifier()
    */
	public R visit(FormalParameter n, A argu) {
		R _ret=null;
		n.f0.accept(this, argu);
		String idName = (String) n.f1.accept(this, argu);
      FP.get(className).get(funName).put(idName, FPcounter++);
		return _ret;
	}

   /** Nothing to do 
    * f0 -> ","
    * f1 -> FormalParameter()
    */
	public R visit(FormalParameterRest n, A argu) {
		R _ret=null;
		n.f0.accept(this, argu);
		n.f1.accept(this, argu);
		return _ret;
	}

   /** DONE
    * f0 -> ArrayType()
    *       | BooleanType()
    *       | IntegerType()
    *       | Identifier()
    */
	public R visit(Type n, A argu) {
		String typeName = (String) n.f0.accept(this, argu);
      return (R) typeName;
	}

   /** DONE
    * f0 -> PublicType()
    *       | PrivateType()
    *       | ProtectedType()
    */
   public R visit(AccessType n, A argu) {
      return n.f0.accept(this, argu);
   }

   /** DONE 
    * f0 -> "int"
    * f1 -> "["
    * f2 -> "]"
    */
   public R visit(ArrayType n, A argu) {
      n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      n.f2.accept(this, argu);
      return (R) "int[]";
   }

   /** DONE
    * f0 -> "boolean"
    */
   public R visit(BooleanType n, A argu) {
      n.f0.accept(this, argu);
      return (R) "boolean";
   }

   /** DONE
    * f0 -> "int"
    */
	public R visit(IntegerType n, A argu) {
		n.f0.accept(this, argu);
		return (R) "int";	
	}

	/** DONE
	* f0 -> "public"
	*/
	public R visit(PublicType n, A argu) {
		n.f0.accept(this, argu);
		return (R) "public";
	}

   /** DONE
    * f0 -> "private"
    */
   public R visit(PrivateType n, A argu) {
      n.f0.accept(this, argu);
      return (R) "private";
   }

   /** DONE
    * f0 -> "protected"
    */
   public R visit(ProtectedType n, A argu) {
      n.f0.accept(this, argu);
      return (R) "protected";
   }

   /** DONE
    * f0 -> Block()
    *       | AssignmentStatement()
    *       | ArrayAssignmentStatement()
    *       | IfStatement()
    *       | WhileStatement()
    *       | PrintStatement()
    */
   public R visit(Statement n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      return _ret;
   }

   /** DONE
    * f0 -> "{"
    * f1 -> ( Statement() )*
    * f2 -> "}"
    */
   public R visit(Block n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      n.f2.accept(this, argu);
      return _ret;
   }

   /** DONE
    * f0 -> Identifier()
    * f1 -> "="
    * f2 -> Expression()
    * f3 -> ";"
    */
   public R visit(AssignmentStatement n, A argu) {
      /* Assignment Statement: Happens in a function so LV then FP then AllGV(use get offset helper) */
      R _ret=null;
      
      String idName = (String) n.f0.accept(this, (A) argu);
      n.f1.accept(this, argu);
      Integer exprResult = (Integer) n.f2.accept(this, argu);
      n.f3.accept(this, argu);
      
      if(LV.get(className).get(funName).containsKey(idName)) {
         int LVOffset = LV.get(className).get(funName).get(idName);
         System.out.println("MOVE TEMP " + LVOffset + " TEMP " + exprResult);
      }
      else if(FP.get(className).get(funName).containsKey(idName)) {
         int FPOffset = FP.get(className).get(funName).get(idName);
         System.out.println("MOVE TEMP " + FPOffset + " TEMP " + exprResult);
      }
      else{
         int GVOffset = getOffset(className, idName);
         System.out.println("HSTORE TEMP 0 " + GVOffset + " TEMP " + exprResult);
      }
      return _ret;
   }

   /** DONE
    * f0 -> Identifier()
    * f1 -> "["
    * f2 -> Expression()
    * f3 -> "]"
    * f4 -> "="
    * f5 -> Expression()
    * f6 -> ";"
    */
   public R visit(ArrayAssignmentStatement n, A argu) {
      String idName = (String) n.f0.accept(this, (A) argu);
      n.f1.accept(this, argu);
      Integer index = (Integer) n.f2.accept(this, argu);
      n.f3.accept(this, argu);
      n.f4.accept(this, argu);
      Integer exprResult = (Integer) n.f5.accept(this, argu);
      n.f6.accept(this, argu);
      
      System.out.println("MOVE TEMP " + tempCounter + " PLUS TEMP " + index + " 1");
      tempCounter++;
      System.out.println("MOVE TEMP " + tempCounter + " TIMES TEMP " + (tempCounter - 1) + " 4");
      tempCounter++;
      int offset = tempCounter - 1;
      
      int baseAddr;
      if(LV.get(className).get(funName).containsKey(idName)) {
         baseAddr = LV.get(className).get(funName).get(idName);
      }
      else if(FP.get(className).get(funName).containsKey(idName)) {
         baseAddr = FP.get(className).get(funName).get(idName);
      }
      else{
         int GVOffset = getOffset(className, idName);
         baseAddr = tempCounter++;
         System.out.println("HLOAD TEMP " + baseAddr + " TEMP 0 " + GVOffset);
      }
      
      System.out.println("MOVE TEMP " + tempCounter + " PLUS TEMP " + baseAddr + " TEMP " + offset);
      tempCounter++;
      System.out.println("HSTORE TEMP " + (tempCounter-1) + " 0 TEMP " + exprResult);
      return (R) null;
   }

   /** Has Nothing to do 
    * f0 -> IfthenElseStatement()
    *       | IfthenStatement()
    */
   public R visit(IfStatement n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      return _ret;
   }

   /** DONE
    * f0 -> "if"
    * f1 -> "("
    * f2 -> Expression()
    * f3 -> ")"
    * f4 -> Statement()
    */
   public R visit(IfthenStatement n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer exprResult = (Integer) n.f2.accept(this, argu);
      int SEnd = labelCounter++;
      System.out.println("CJUMP TEMP " + exprResult + " L" + SEnd);
      n.f3.accept(this, argu);
      n.f4.accept(this, argu);
      System.out.println("L"+SEnd+"\nNOOP");
      labelCounter++;
      return _ret;
   }

   /** DONE
    * f0 -> "if"
    * f1 -> "("
    * f2 -> Expression()
    * f3 -> ")"
    * f4 -> Statement()
    * f5 -> "else"
    * f6 -> Statement()
    */
   public R visit(IfthenElseStatement n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer exprResult = (Integer) n.f2.accept(this, argu);
      int S2Begin = labelCounter++;
      int S2Next = labelCounter++;
      System.out.println("CJUMP TEMP " + exprResult + " L" + S2Begin);
      n.f3.accept(this, argu);
      n.f4.accept(this, argu);
      System.out.println("JUMP L" + S2Next);
      n.f5.accept(this, argu);
      System.out.println("L"+S2Begin+"\nNOOP");
      n.f6.accept(this, argu);
      System.out.println("L"+S2Next+"\nNOOP");
      return _ret;
   }

   /** DONE
    * f0 -> "while"
    * f1 -> "("
    * f2 -> Expression()
    * f3 -> ")"
    * f4 -> Statement()
    */
   public R visit(WhileStatement n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      int checkCond = labelCounter++;
      int exitLoop = labelCounter++;
      System.out.println("L"+checkCond+"\nNOOP");
      Integer exprResult = (Integer) n.f2.accept(this, argu);
      System.out.println("CJUMP TEMP " + exprResult + " L" + exitLoop);
      n.f3.accept(this, argu);
      n.f4.accept(this, argu);
      System.out.println("JUMP L"+checkCond);
      System.out.println("L"+exitLoop+"\nNOOP");
      return _ret;
   }

   /** DONE 
    * f0 -> "System.out.println"
    * f1 -> "("
    * f2 -> Expression()
    * f3 -> ")"
    * f4 -> ";"
    */
   public R visit(PrintStatement n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer exprResult = (Integer) n.f2.accept(this, argu);
      System.out.println("PRINT TEMP " + exprResult);
      n.f3.accept(this, argu);
      n.f4.accept(this, argu);
      return _ret;
   }

   /** DONE
    * f0 -> OrExpression()
    *       | AndExpression()
    *       | CompareExpression()
    *       | neqExpression()
    *       | PlusExpression()
    *       | MinusExpression()
    *       | TimesExpression()
    *       | DivExpression()
    *       | ArrayLookup()
    *       | ArrayLength()
    *       | MessageSend()
    *       | TernaryExpression()
    *       | PrimaryExpression()
    */
   public R visit(Expression n, A argu) {
      return n.f0.accept(this, argu);
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "&&"
    * f2 -> PrimaryExpression()
    */
   public R visit(AndExpression n, A argu) {
      int BEnd = labelCounter++;
      int res = tempCounter++;
      
      System.out.println("MOVE TEMP " + res + " 0");

      Integer LHS =  (Integer) n.f0.accept(this, argu);
      System.out.println("CJUMP TEMP " + LHS + " L" + BEnd);
      n.f1.accept(this, argu);
      Integer RHS = (Integer) n.f2.accept(this, argu);
      System.out.println("CJUMP TEMP " + RHS + " L" + BEnd);
      System.out.println("MOVE TEMP " + res + " 1");
      
      System.out.println("L" + BEnd + "\nNOOP");
      
      return (R)(Integer)res;
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "||"
    * f2 -> PrimaryExpression()
    */
   public R visit(OrExpression n, A argu) {
      int BFalse = labelCounter++;
      int BEnd = labelCounter++;
      int res = tempCounter++;
      
      System.out.println("MOVE TEMP " + res + " 0");

      Integer LHS =  (Integer) n.f0.accept(this, argu);
      System.out.println("CJUMP TEMP " + LHS + " L" + BFalse);
      System.out.println("MOVE TEMP " + res + " 1");
      System.out.println("JUMP L" + BEnd);
      
      n.f1.accept(this, argu);
      
      System.out.println("L"+BFalse+"\nNOOP");
      Integer RHS = (Integer) n.f2.accept(this, argu);
      System.out.println("CJUMP TEMP " + RHS + " L" + BEnd);
      System.out.println("MOVE TEMP " + res + " 1");
      
      System.out.println("L" + BEnd + "\nNOOP");
      return (R) (Integer) res;
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "<="
    * f2 -> PrimaryExpression()
    */
   public R visit(CompareExpression n, A argu) {
      Integer LHS =  (Integer) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer RHS = (Integer) n.f2.accept(this, argu);
      System.out.println("MOVE TEMP " + tempCounter + " LE TEMP " + LHS + " TEMP " + RHS);
      tempCounter++;
      return (R) (Integer) (tempCounter-1);
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "!="
    * f2 -> PrimaryExpression()
    */
   public R visit(neqExpression n, A argu) {
      Integer LHS =  (Integer) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer RHS = (Integer) n.f2.accept(this, argu);
      System.out.println("MOVE TEMP " + tempCounter + " NE TEMP " + LHS + " TEMP " + RHS);
      tempCounter++;
      return (R)(Integer)(tempCounter-1);
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "+"
    * f2 -> PrimaryExpression()
    */
   public R visit(PlusExpression n, A argu) {
      Integer LHS =  (Integer) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer RHS = (Integer) n.f2.accept(this, argu);
      System.out.println("MOVE TEMP " + tempCounter + " PLUS TEMP " + LHS + " TEMP " + RHS);
      tempCounter++;
      return (R)(Integer)(tempCounter-1);
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "-"
    * f2 -> PrimaryExpression()
    */
   public R visit(MinusExpression n, A argu) {
      Integer LHS =  (Integer) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer RHS = (Integer) n.f2.accept(this, argu);
      System.out.println("MOVE TEMP " + tempCounter + " MINUS TEMP " + LHS + " TEMP " + RHS);
      tempCounter++;
      return (R)(Integer)(tempCounter-1);
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "*"
    * f2 -> PrimaryExpression()
    */
   public R visit(TimesExpression n, A argu) {
      Integer LHS =  (Integer) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer RHS = (Integer) n.f2.accept(this, argu);
      System.out.println("MOVE TEMP " + tempCounter + " TIMES TEMP " + LHS + " TEMP " + RHS);
      tempCounter++;
      return (R)(Integer)(tempCounter-1);
   }

   /** DONE 
    * f0 -> PrimaryExpression()
    * f1 -> "/"
    * f2 -> PrimaryExpression()
    */
   public R visit(DivExpression n, A argu) {
      Integer LHS =  (Integer) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer RHS = (Integer) n.f2.accept(this, argu);
      System.out.println("MOVE TEMP " + tempCounter + " DIV TEMP " + LHS + " TEMP " + RHS);
      tempCounter++;
      return (R)(Integer)(tempCounter-1);
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "["
    * f2 -> PrimaryExpression()
    * f3 -> "]"
    */
   public R visit(ArrayLookup n, A argu) {
      Integer LHS  = (Integer) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      Integer RHS = (Integer) n.f2.accept(this, argu);
      n.f3.accept(this, argu);
      
      System.out.println("MOVE TEMP " + tempCounter + " PLUS TEMP " + RHS + " 1");
      tempCounter++;
      System.out.println("MOVE TEMP " + tempCounter + " TIMES TEMP " + (tempCounter - 1) + " 4"); // At index zero length is stored
      tempCounter++;
      System.out.println("MOVE TEMP " + tempCounter + " PLUS TEMP " + LHS + " TEMP " + (tempCounter - 1));
      tempCounter++;
      System.out.println("HLOAD TEMP " + tempCounter + " TEMP " + (tempCounter - 1) + " 0");
      tempCounter++;
      return (R)(Integer)(tempCounter-1);
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "."
    * f2 -> "length"
    */
   public R visit(ArrayLength n, A argu) {
      Integer LHS = (Integer) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      n.f2.accept(this, argu);

      System.out.println("HLOAD TEMP " + tempCounter + " TEMP " + LHS + " 0");
      tempCounter++;
      return (R)(Integer)(tempCounter-1);
   }

   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "?"
    * f2 -> PrimaryExpression()
    * f3 -> ":"
    * f4 -> PrimaryExpression()
    */
    public R visit(TernaryExpression n, A argu) {
      int BFalse = labelCounter++;
      int BEnd = labelCounter++;
      Integer cond = (Integer) n.f0.accept(this, argu);
      System.out.println("CJUMP TEMP " + cond + " L" + BFalse);
      n.f1.accept(this, argu);
      
      int res = tempCounter++;
      // BTrue:
      Integer arg1 = (Integer) n.f2.accept(this, argu);
      n.f3.accept(this, argu);
      System.out.println("MOVE TEMP " + res + " TEMP " + arg1);
      System.out.println("JUMP L"+BEnd);
      
      // BFalse:
      System.out.println("L"+BFalse+"\nNOOP");
      Integer arg2 = (Integer) n.f4.accept(this, argu);
      System.out.println("MOVE TEMP " + res + " TEMP " + arg2);

      // BEnd:
      System.out.println("L"+BEnd+"\nNOOP");
      
      return (R) (Integer) res;
   }
   
   /** DONE
    * f0 -> PrimaryExpression()
    * f1 -> "."
    * f2 -> Identifier()
    * f3 -> "("
    * f4 -> ( ExpressionList() )?
    * f5 -> ")"
    */
	public R visit(MessageSend n, A argu) {
      Integer classRecordReg = (Integer) n.f0.accept(this, argu);
		n.f1.accept(this, argu);
      String methodName = (String) n.f2.accept(this, argu);
      int ret = tempCounter++;
		n.f3.accept(this, argu);
      ArrayList<Integer> exprList = new ArrayList<Integer>();
		n.f4.accept(this, (A) exprList);
      
      // First find the offset of the function
      int FTOffset = 4 * FunTable.get(methodName);
      // Find the addr of Function table in IR
      System.out.println("HLOAD TEMP " + tempCounter + " TEMP " + classRecordReg + " 0");
      tempCounter++;
      // Then find the label address using the function table
      System.out.println("HLOAD TEMP " + tempCounter + " TEMP " + (tempCounter - 1) + " " + FTOffset);
      tempCounter++;
      
      System.out.print("MOVE TEMP " + ret + " CALL TEMP " + (tempCounter - 1) + " ( TEMP " + classRecordReg);
      for (int i = 0; i < exprList.size(); i++) System.out.print(" TEMP " + exprList.get(i));
      System.out.println(" )");
		n.f5.accept(this, argu);
      return (R) (Integer) ret;
	}

   /** Has Nothing to do
    * f0 -> Expression()
    * f1 -> ( ExpressionRest() )*
    */
   public R visit(ExpressionList n, A argu) {
      R _ret=null;
      ArrayList<Integer> exprList = (ArrayList<Integer>) argu;
      exprList.add((Integer) n.f0.accept(this, argu));
      n.f1.accept(this, (A) exprList);
      return _ret;
   }

   /** DONE
    * f0 -> ","
    * f1 -> Expression()
    */
   public R visit(ExpressionRest n, A argu) {
      R _ret=null;
      ArrayList<Integer> exprList = (ArrayList<Integer>) argu;
      n.f0.accept(this, argu);
      exprList.add((Integer) n.f1.accept(this, argu));
      return _ret;
   }

   boolean isPriExpr = false;
   /** DONE
    * f0 -> IntegerLiteral()
    *       | TrueLiteral()
    *       | FalseLiteral()
    *       | Identifier()
    *       | ThisExpression()
    *       | ArrayAllocationExpression()
    *       | AllocationExpression()
    *       | NotExpression()
    *       | BracketExpression()
    */
   public R visit(PrimaryExpression n, A argu) {
      boolean store = isPriExpr; isPriExpr = true;
      R x = n.f0.accept(this, argu);
      isPriExpr = store;
      return x;
   }

   /** DONE
    * f0 -> <INTEGER_LITERAL>
    */
   public R visit(IntegerLiteral n, A argu) {
      String intValStr = (String) n.f0.accept(this, argu);
      Integer intVal = Integer.valueOf(intValStr);
      System.out.println("MOVE TEMP " + tempCounter + " " + intVal);
      tempCounter++;
      return (R) (Integer) (tempCounter - 1);
   }

   /** DONE
    * f0 -> "true"
    */
   public R visit(TrueLiteral n, A argu) {
      n.f0.accept(this, argu);
      int intVal = 1;
      System.out.println("MOVE TEMP " + tempCounter + " " + intVal);
      tempCounter++;
      return (R) (Integer) (tempCounter - 1);
   }

	/** DONE 
	* f0 -> "false"
	*/
	public R visit(FalseLiteral n, A argu) {
		n.f0.accept(this, argu);
      int intVal = 0;
      System.out.println("MOVE TEMP " + tempCounter + " " + intVal);
      tempCounter++;
      return (R) (Integer) (tempCounter - 1);
	}

	/** DONE
	* f0 -> <IDENTIFIER>
	*/
	public R visit(Identifier n, A argu) {
		String idName = (String) n.f0.accept(this, argu);
      if(isPriExpr){ // This stores the base of class record of id in new TEMP
         if(LV.get(className).get(funName).containsKey(idName)) {
            int LVOffset = LV.get(className).get(funName).get(idName);
            System.out.println("MOVE TEMP " + tempCounter + " TEMP " + LVOffset);
            tempCounter++;
            return (R) (Integer) (tempCounter - 1);
         }
         else if(FP.get(className).get(funName).containsKey(idName)) {
            int FPOffset = FP.get(className).get(funName).get(idName);
            System.out.println("MOVE TEMP " + tempCounter + " TEMP " + FPOffset);
            tempCounter++;
            return (R) (Integer) (tempCounter - 1);
         }
         else{
            int GVOffset = getOffset(className, idName);
            System.out.println("HLOAD TEMP " + tempCounter + " TEMP 0 " + GVOffset); 
            tempCounter++;
            return (R) (Integer) (tempCounter - 1);
         }
      }
      return (R) idName;
	}

	/** DONE
	* f0 -> "this"
	*/
	public R visit(ThisExpression n, A argu) {
		n.f0.accept(this, argu);
      System.out.println("MOVE TEMP " + tempCounter + " TEMP 0");
      tempCounter++;
		return (R) (Integer) (tempCounter - 1);
	}

   /** DONE
    * f0 -> "new"
    * f1 -> "int"
    * f2 -> "["
    * f3 -> Expression()
    * f4 -> "]"
    */
	public R visit(ArrayAllocationExpression n, A argu) {
		n.f0.accept(this, argu);
		n.f1.accept(this, argu);
		n.f2.accept(this, argu);
      boolean storeisPri = isPriExpr; isPriExpr = false;
		Integer exprResult = (Integer) n.f3.accept(this, argu);
      isPriExpr = storeisPri;
		n.f4.accept(this, argu);
      
      // Alloacate len+1 and store length in index 0
      System.out.println("MOVE TEMP " + tempCounter + " PLUS TEMP " + exprResult + " 1");
      tempCounter++;
      System.out.println("MOVE TEMP " + tempCounter + " TIMES TEMP " + (tempCounter-1) + " 4");
      tempCounter++;
      System.out.println("MOVE TEMP " + tempCounter + " HALLOCATE TEMP " + (tempCounter-1));
      System.out.println("HSTORE TEMP " + tempCounter + " 0 TEMP " + exprResult);
      tempCounter++;
		return (R) (Integer) (tempCounter - 1);
	}

   /** DONE
    * f0 -> "new"
    * f1 -> Identifier()
    * f2 -> "("
    * f3 -> ")"
    */
	public R visit(AllocationExpression n, A argu) {
		n.f0.accept(this, argu);
      boolean store = isPriExpr; isPriExpr = false;
		String idName = (String) n.f1.accept(this, argu);
      isPriExpr = store;
		n.f2.accept(this, argu);
		n.f3.accept(this, argu);
      
      int recordSize = 4 * (countAllGV(idName) + 1); // 1 is for function table
      System.out.println("MOVE TEMP " + tempCounter + " HALLOCATE " + recordSize);
      int baseCR = tempCounter;
      tempCounter++;
      System.out.println("MOVE TEMP " + tempCounter + " HALLOCATE " + (FunTable.size() * 4 + 1) );
      int baseFT = tempCounter;
      tempCounter++;
      System.out.println("HSTORE TEMP " + baseCR + " 0 TEMP " + (tempCounter - 1));
      // Now we have to fill the function table
      fillFunTable(idName, baseFT);
		return (R) (Integer) (baseCR);
	}

   /** DONE
    * f0 -> "!"
    * f1 -> Expression()
    */
	public R visit(NotExpression n, A argu) {
		n.f0.accept(this, argu);
      boolean store = isPriExpr; isPriExpr = false;
      int exprResult = (Integer) n.f1.accept(this, argu);
      isPriExpr = store;
      System.out.println("MOVE TEMP " + tempCounter + " NE TEMP " + exprResult + " 1");
      tempCounter++;
		return (R) (Integer) (tempCounter - 1);
	}

   /** DONE
    * f0 -> "("
    * f1 -> Expression()
    * f2 -> ")"
    */
   public R visit(BracketExpression n, A argu) {
      n.f0.accept(this, argu);
      boolean store = isPriExpr; isPriExpr = false;
      Integer exprResult = (Integer) n.f1.accept(this, argu);
      isPriExpr = store;
      n.f2.accept(this, argu);
      return (R) exprResult;
   }

   /** WASTE
    * f0 -> Identifier()
    * f1 -> ( IdentifierRest() )*
    */
	public R visit(IdentifierList n, A argu) {
		R _ret=null;
		n.f0.accept(this, argu);
		n.f1.accept(this, argu);
		return _ret;
	}

   /** WASTE
    * f0 -> ","
    * f1 -> Identifier()
    */
	public R visit(IdentifierRest n, A argu) {
		R _ret=null;
		n.f0.accept(this, argu);
		n.f1.accept(this, argu);
		return _ret;
	}

}
