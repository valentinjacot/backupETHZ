package inheritance;

import java.util.ArrayList;

// inheritance based stack implementation
public class Stack1<T> extends ArrayList<T> {
	private static final long serialVersionUID = -536111595965183422L;

	public Stack1() { }

	public Object push(T item) {
		add(item);
		return item;
	}
}