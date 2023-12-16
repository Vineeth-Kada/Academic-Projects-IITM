import syntaxtree.*;
import visitor.*;
import java.util.*;

public class A3 {
   public static void main(String [] args) {
      try {
         HashMap<String, ArrayList<String>> funs = new HashMap<>();
         HashMap<String, HashMap<String, Integer>> LV = new HashMap<>();
         HashMap<String, String> parent = new HashMap<>();
         HashMap<String, Integer> FunTable = new HashMap<>();
         Node root = new MiniJavaParser(System.in).Goal();
         // System.out.println("Program parsed successfully");
         root.accept(new GJDepthFirst1(funs, LV, parent, FunTable), null); // First Parse
         root.accept(new GJDepthFirst2(funs, LV, parent, FunTable), null); // Second Parse
      }
      catch (ParseException e) {
         System.out.println(e.toString());
      }
   }
}
