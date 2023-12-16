import syntaxtree.*;
import visitor.*;
import java.util.*;

public class A2 {
   public static void main(String [] args) {
      try {
         HashMap<String, HashMap<String, ArrayList<String>>> funs = new HashMap<>();
         HashMap<String, HashMap<String, String>> LV = new HashMap<>();
         HashMap<String, String> parent = new HashMap<>();
         Node root = new MiniJavaParser(System.in).Goal();
         // System.out.println("Program parsed successfully");
         root.accept(new GJDepthFirst1(funs, LV, parent), null); // First Parse
         root.accept(new GJDepthFirst2(funs, LV, parent), null); // Second Parse
      }
      catch (ParseException e) {
         System.out.println(e.toString());
      }
   }
}
