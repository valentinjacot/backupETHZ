package patterns.clone.alias;

import java.io.Serializable;

import patterns.clone.comparison.util.SerializableClone;

//This program demonstrates that reflective cloning and serialization can solve this problem.
public class TestDiamond1 {

	@SuppressWarnings("serial")
	static class Node implements Serializable {
		private Node left, right;
		private int val;

		@SuppressWarnings("unused")
		private Node(){} // used for reflective clone
		public Node(int val, Node left, Node right) {
			this.val = val;
			this.left = left;
			this.right = right;
		}

		public Node getLeft() {  return left; }
		public Node getRight() { return right; }
		public int getVal() { return val; }
	}

	public static void main(String[] args) {
		Node n4 = new Node(4, null, null);
		Node n2 = new Node(2, null, n4);
		Node n3 = new Node(3, n4, null);
		Node n1 = new Node(1, n2, n3);
		
		System.out.println(n1.getLeft().getRight());
		System.out.println(n1.getRight().getLeft());
		System.out.println(n1.getLeft().getRight() == n1.getRight().getLeft());

		Node c = null;
		c = (Node) SerializableClone.clone(n1);
//		c = (Node) ReflectiveClone.clone(n1);
//
//		Cloner cloner=new Cloner();
//		c = cloner.deepClone(n1);
	
		System.out.println(c.getLeft().getRight());
		System.out.println(c.getRight().getLeft());
		System.out.println(c.getLeft().getRight() == c.getRight().getLeft());
	}
}
