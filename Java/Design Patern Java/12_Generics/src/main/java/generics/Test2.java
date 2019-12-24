package generics;

public class Test2 {
   static class A {}
   static class B extends A  {}

   public static void main(String[] args) {
      B[] arrayB = new B[1];
      A[] arrayA = arrayB;
      arrayA[0] = new A();
   }
}
