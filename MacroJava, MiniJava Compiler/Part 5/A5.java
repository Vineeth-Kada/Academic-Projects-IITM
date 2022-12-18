import syntaxtree.*;
import visitor.*;
import java.util.*;

public class A5 {
   public static void main(String [] args) {
      try {
         Node root = new MiniRAParser(System.in).Goal();
         // System.out.println("Program parsed successfully");
         root.accept(new GJDepthFirst(), null);
      }
      catch (ParseException e) {
         System.out.println(e.toString());
      }
   }
}
