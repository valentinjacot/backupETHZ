package inheritance;

import java.util.ArrayList;

// composition based stack implementation
public class Stack2<T> {
	public Stack2() { }
	private ArrayList<T> list = new ArrayList<>();

	public Object push(T item) {
		list.add(item);
		return item;
	}
}