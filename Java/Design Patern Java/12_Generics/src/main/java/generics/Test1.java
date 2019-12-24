package generics;

import java.util.ArrayList;

@SuppressWarnings("unused")
public class Test1 {
   static class A {}
   static class B extends A {}

   public static void main(String[] args) {
      ArrayList<B> listB = new ArrayList<B>();
//      ArrayList<A> listA = listB;
   }
}
