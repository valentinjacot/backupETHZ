package patterns.clone.alias;

import java.io.Serializable;

import patterns.clone.comparison.util.SerializableClone;

// This program demonstrates that reflective cloning and serialization can solve this problem.
public class TestCycle1 {

	@SuppressWarnings("serial")
	static class Node implements Serializable {
		Node next;
		int val;

		@SuppressWarnings("unused")
		private Node() { } // used for reflective clone
		public Node(int val, Node next) {
			this.val = val;
			this.next = next;
		}
	}

	public static void main(String[] args) {
		Node n1 = new Node(1, null);
		Node n2 = new Node(2, n1);
		Node n3 = new Node(3, n2);
		n1.next = n3;
		// n3 -> n2 -> n1 -> n3 -> ....

		Node c = null;
		c = (Node) SerializableClone.clone(n1);
//		c = (Node) ReflectiveClone.clone(n1);
//		
//		Cloner cloner=new Cloner();
//		c = cloner.deepClone(n1);
	
		System.out.println(c);
		System.out.println(c.next);
		System.out.println(c.next.next);
		System.out.println(c.next.next.next);
		System.out.println(c == c.next.next.next);
	}
}
