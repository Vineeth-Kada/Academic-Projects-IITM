//
// Generated by JTB 1.3.2
//

package visitor;
import syntaxtree.*;
import java.util.*;

/**
 * Provides default methods which visit each node in the tree in depth-first
 * order.  Your visitors may extend this class.
 */
public class GJDepthFirst3<R,A> implements GJVisitor<R,A> {
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

   public R visit(NodeToken n, A argu) { return (R) (String)n.tokenImage; }

   //
   // User-generated visitor methods below
   //
   HashMap<String, HashMap<Integer, String>> register;
   HashMap<String, Integer> maxArgCnt;
   HashMap<String, Integer> spilledTemp;
   HashMap<String, HashMap<Integer, Integer>> location;
   public GJDepthFirst3(      HashMap<String, HashMap<Integer, String>> __register,
                              HashMap<String, Integer> __maxArgCnt,
                              HashMap<String, Integer> __spilledTemp,
                              HashMap<String, HashMap<Integer, Integer>> __location ){
		register = __register;
      maxArgCnt = __maxArgCnt;
      spilledTemp = __spilledTemp;
      location = __location;
	}
   int counter = 0;
   int argCnt;
   String procLabel;
   int tBase, sBase, fcall;
   HashMap<Integer, String> reg;
   HashMap<Integer, Integer> loc;
   public void procBegin(){
      int mx = maxArgCnt.get(procLabel);
      int stackUsed = spilledTemp.get(procLabel);
      fcall = (mx == -1 ? 0 : 10);
      if(procLabel == "MAIN") tBase = stackUsed;
      else{
         sBase = stackUsed;
         tBase = stackUsed + 8;
      }
      int stackSize = stackUsed + fcall + (procLabel == "MAIN" ? 0 : 8);
      if(mx == -1) mx = 0;
      System.out.println(procLabel + " [" + argCnt + "] [" + stackSize + "] [" + mx + "]");
      if(procLabel != "MAIN"){
         for(int i=0; i<8; i++) { int loc = sBase + i; System.out.println("\tASTORE SPILLEDARG " + loc + " s" + i); }
      }
      reg = register.get(procLabel);
      loc = location.get(procLabel);
      for(int i=0; i<Math.min(4, argCnt); i++){
         if(reg.containsKey(i)){
            System.out.println("\tMOVE " + reg.get(i) + " a" + i);
         }
         else{
            System.out.println("\tMOVE v0 a" + i);
            processLoad(i);
         }
      }
   }
   public void procEnd(){
      if(procLabel != "MAIN"){
         for(int i=0; i<8; i++) { int loc = sBase + i; System.out.println("\tALOAD s" + i + " SPILLEDARG " + loc); }
      }
      System.out.println("\tEND");
   }

   // DONE
   public String process(int temp, boolean isZeroUsed){  // v0/v1
      if(reg.containsKey(temp)) return reg.get(temp);
      else{
         System.out.println("\tALOAD " + (isZeroUsed ? "v1 " : "v0 ") + "SPILLEDARG " + loc.get(temp));
         return (isZeroUsed ? "v1" : "v0");
      }
   }
   
   // DONE
   public void processLoad(int temp){  // value is in v0
      if(reg.containsKey(temp)) System.out.println("\tMOVE " + reg.get(temp) + " v0");
      else if(loc.containsKey(temp)) System.out.println("\tASTORE SPILLEDARG " + loc.get(temp) + " v0");
      else System.out.println("\tMOVE v1 v0");
   }
   
   /** DONE
    * f0 -> "MAIN"
    * f1 -> StmtList()
    * f2 -> "\tEND"
    * f3 -> ( Procedure() )*
    * f4 -> <EOF>
    */
   public R visit(Goal n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      procLabel = "MAIN";
      argCnt = 0;
      procBegin();
      n.f1.accept(this, argu);
      n.f2.accept(this, argu);
      procEnd();
      n.f3.accept(this, argu);
      n.f4.accept(this, argu);
      return _ret;
   }

   /** DONE
    * f0 -> ( ( Label() )? Stmt() )*
    */
   boolean printLabel = false;
   public R visit(StmtList n, A argu) {
      R _ret=null;
      printLabel = true;
      n.f0.accept(this, argu);
      printLabel = false;
      return _ret;
   }

   /** DONE
    * f0 -> Label()
    * f1 -> "["
    * f2 -> IntegerLiteral()
    * f3 -> "]"
    * f4 -> StmtExp()
    */
   public R visit(Procedure n, A argu) {
      R _ret=null;
      procLabel = (String) n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      argCnt = (Integer) n.f2.accept(this, argu);
      procBegin();
      n.f3.accept(this, argu);
      n.f4.accept(this, argu);
      procEnd();
      return _ret;
   }

   /** DONE
    * f0 -> NoOpStmt()
    *       | ErrorStmt()
    *       | CJumpStmt()
    *       | JumpStmt()
    *       | HStoreStmt()
    *       | HLoadStmt()
    *       | MoveStmt()
    *       | PrintStmt()
    */
   public R visit(Stmt n, A argu) {
      R _ret=null;
      printLabel = false;
      n.f0.accept(this, argu);
      printLabel = true;
      return _ret;
   }

   /** DONE
    * f0 -> "\tNOOP"
    */
   public R visit(NoOpStmt n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      System.out.println("\tNOOP");
      return _ret;
   }

   /** DONE
    * f0 -> "\tERROR"
    */
   public R visit(ErrorStmt n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      System.out.println("\tERROR");
      return _ret;
   }

   /** DONE
    * f0 -> "\tCJUMP"
    * f1 -> Temp()
    * f2 -> Label()
    */
   public R visit(CJumpStmt n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      Integer temp = (Integer) n.f1.accept(this, argu);
      String Label = (String) n.f2.accept(this, argu);
      String regForTemp = process(temp, false);
      System.out.println("\tCJUMP " + regForTemp + " " + Label);
      return _ret;
   }

   /** DONE
    * f0 -> "\tJUMP"
    * f1 -> Label()
    */
   public R visit(JumpStmt n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      String Label = (String) n.f1.accept(this, argu);
      System.out.println("\tJUMP " + Label);
      return _ret;
   }

   /** DONE
    * f0 -> "\tHSTORE"
    * f1 -> Temp()
    * f2 -> IntegerLiteral()
    * f3 -> Temp()
    */
   public R visit(HStoreStmt n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      Integer temp1 = (Integer) n.f1.accept(this, argu);
      Integer offset = (Integer) n.f2.accept(this, argu);
      Integer temp2 = (Integer) n.f3.accept(this, argu);
      String reg1 = process(temp1, false);
      String reg2 = process(temp2, true);
      System.out.println("\tHSTORE " + reg1 + " " + offset + " " + reg2);
      return _ret;
   }

   /** DONE
    * f0 -> "\tHLOAD"
    * f1 -> Temp()
    * f2 -> Temp()
    * f3 -> IntegerLiteral()
    */
   public R visit(HLoadStmt n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      Integer temp1 = (Integer) n.f1.accept(this, argu);
      Integer temp2 = (Integer) n.f2.accept(this, argu);
      Integer offset = (Integer) n.f3.accept(this, argu);
      String reg2 = process(temp2, true);
      if(reg.containsKey(temp1)){
         String reg1 = process(temp1, false);
         System.out.println("\tHLOAD " + reg1 + " " + reg2 + " " + offset);
      }
      else{
         System.out.println("\tHLOAD v0 " + reg2 + " " + offset);
         processLoad(temp1);
      }
      return _ret;
   }

   /** DONE
    * f0 -> "BEGIN"
    * f1 -> StmtList()
    * f2 -> "RETURN"
    * f3 -> SimpleExp()
    * f4 -> "\tEND"
    */
    public R visit(StmtExp n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      n.f1.accept(this, argu);
      n.f2.accept(this, argu);
      String retReg = (String) n.f3.accept(this, argu);
      System.out.println("\tMOVE v0 " + retReg);
      n.f4.accept(this, argu);
      return _ret;
   }
   
   /** DONE
    * f0 -> "\tPRINT"
    * f1 -> SimpleExp()
    */
    public R visit(PrintStmt n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      String arg1 = (String) n.f1.accept(this, argu);
      System.out.println("\tPRINT " + arg1);
      return _ret;
   }
   
   /** DONE
    * f0 -> "\tMOVE"
    * f1 -> Temp()
    * f2 -> Exp()
    */
   public R visit(MoveStmt n, A argu) {
      R _ret=null;
      n.f0.accept(this, argu);
      Integer temp = (Integer) n.f1.accept(this, argu);
      n.f2.accept(this, (A)temp);
      return _ret;
   }

   /** DONE
    * f0 -> \tCALL()
    *       | HAllocate()
    *       | BinOp()
    *       | SimpleExp()
    */
   boolean isSimpleEinExpr = false;
   public R visit(Exp n, A argu) {
      isSimpleEinExpr = true;
      n.f0.accept(this, argu);
      isSimpleEinExpr = false;
      return (R) null;
   }

   /**
    * f0 -> "\tCALL"
    * f1 -> SimpleExp()
    * f2 -> "("
    * f3 -> ( Temp() )*
    * f4 -> ")"
    */
   int argId = 0;
   boolean inCall = false;
   public R visit(Call n, A argu) {
      for(int i=0; i<10; i++) { int loc = tBase + i; System.out.println("\tASTORE SPILLEDARG " + loc + " t" + i); }
      
      R _ret=null;
      isSimpleEinExpr = false;
      n.f0.accept(this, argu);
      String SExpr = (String) n.f1.accept(this, argu);
      n.f2.accept(this, argu);
      argId = 0; inCall = true;
      n.f3.accept(this, argu);
      inCall = false;
      n.f4.accept(this, argu);
      System.out.println("\tCALL " + SExpr);
      for(int i=0; i<10; i++) { int loc = tBase + i; System.out.println("\tALOAD t" + i + " SPILLEDARG " + loc); }
      
      processLoad((Integer) argu);
      return _ret;
   }

   /** DONE
    * f0 -> "HALLOCATE"
    * f1 -> SimpleExp()
    */
   public R visit(HAllocate n, A argu) {
      isSimpleEinExpr = false;
      R _ret=null;
      n.f0.accept(this, argu);
      String reg1 = (String) n.f1.accept(this, argu);
      if(reg.containsKey(argu)) System.out.println("\tMOVE " + reg.get(argu) + " " + "HALLOCATE " + reg1);
      else{
         System.out.println("\tMOVE v0 HALLOCATE " + reg1);
         processLoad((Integer) argu);
      }
      return _ret;
   }

   /** DONE
    * f0 -> Operator()
    * f1 -> Temp()
    * f2 -> SimpleExp()
    */
   public R visit(BinOp n, A argu) {
      isSimpleEinExpr = false;
      R _ret=null;
      String opCode = (String) n.f0.accept(this, argu);
      Integer op1 = (Integer) n.f1.accept(this, argu);
      String reg1 = process(op1, false);
      String reg2 = (String) n.f2.accept(this, argu);
      if(reg.containsKey(argu)) System.out.println("\tMOVE " + reg.get(argu) + " " + opCode + " " + reg1 + " " + reg2);
      else{
         System.out.println("\tMOVE v0" + opCode + " " + reg1 + " " + reg2);
         processLoad((Integer) argu);
      }
      return _ret;
   }
   
   /** DONE - Returns the register / Label - If stored in stack uses v1 not v0
    * f0 -> Temp()
    *       | IntegerLiteral()
    *       | Label()
    */
    boolean SimpleExp = false;
    public R visit(SimpleExp n, A argu) {
      SimpleExp = true;
      String Store = (String) n.f0.accept(this, argu);
      SimpleExp = false;
      if(isSimpleEinExpr){
         int temp = (Integer) argu;
         if(reg.containsKey(temp)){
            String regTemp = process(temp, false);
            System.out.println("\tMOVE " + regTemp + " " + Store);
         }
         else{
            System.out.println("\tMOVE v0 " + Store);
            processLoad(temp);
         }
      }
      return (R) Store;
   }

   /** DONE
    * f0 -> "LE"
    *       | "NE"
    *       | "PLUS"
    *       | "MINUS"
    *       | "TIMES"
    *       | "DIV"
    */
   public R visit(Operator n, A argu) {
      return n.f0.accept(this, argu);
   }

   /** DONE
    * f0 -> "TEMP"
    * f1 -> IntegerLiteral()
    */
   public R visit(Temp n, A argu) {
      n.f0.accept(this, argu);
      boolean temppp = SimpleExp; SimpleExp = false;
      Integer Temp = (Integer) n.f1.accept(this, argu);
      SimpleExp = temppp;
      if(SimpleExp){
         String tempReg = process(Temp, true);
         return (R)tempReg;
      }
      
      if(inCall){
         String TempReg = process(Temp, false);
         if(argId < 4) System.out.println("\tMOVE a" + argId + " " + TempReg);
         else System.out.println("\tPASSARG " + (argId+1-4) + " " + TempReg);
         argId++;   
      }
      
      return (R)Temp;
   }

   /** DONE
    * f0 -> <INTEGER_LITERAL>
    */
   public R visit(IntegerLiteral n, A argu) {
      String val = (String) n.f0.accept(this, argu);
      if(SimpleExp) return (R) val;
      else return (R) Integer.valueOf(val);
   }

   /** DONE
    * f0 -> <IDENTIFIER>
    */
   public R visit(Label n, A argu) {
      String local_label = (String) n.f0.accept(this, argu);
      if(printLabel) {System.out.println(local_label);}
      return (R) local_label;
   }

}