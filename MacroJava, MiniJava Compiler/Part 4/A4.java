import syntaxtree.*;
import visitor.*;
import java.util.*;

public class A4 {
   public static void main(String [] args) {
      try {
         Node root = new microIRParser(System.in).Goal();
         // System.out.println("Program parsed successfully");
         HashMap<String, HashMap<String, Integer>> labelLine = new HashMap<>();
         root.accept(new GJDepthFirst1(labelLine), null);
         HashMap<String, HashMap<Integer, String>> register = new HashMap<>();
         HashMap<String, Integer> maxArgCnt = new HashMap<>();
         HashMap<String, Integer> spilledTemp = new HashMap<>();
         HashMap<String, HashMap<Integer, Integer>> location = new HashMap<>();
         root.accept(new GJDepthFirst2(labelLine, register, maxArgCnt, spilledTemp, location), null); // First Parse
         root.accept(new GJDepthFirst3(register, maxArgCnt, spilledTemp, location), null); // Second Parse
      }
      catch (ParseException e) {
         System.out.println(e.toString());
      }
   }
}
