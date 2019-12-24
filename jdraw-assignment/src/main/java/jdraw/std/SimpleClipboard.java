package jdraw.std;

import java.util.LinkedList;
import java.util.List;

import jdraw.framework.Figure;

public final class SimpleClipboard {
	public SimpleClipboard() {}
	private static List<Figure> figures =new LinkedList<>();
	
	public static void add (Figure f) {
		figures.add(f);
	}
	
	public static List<Figure> get(){
		return figures;
	}
	public static void clear() {
		figures.clear();
	}
}
