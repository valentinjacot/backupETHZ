package patterns.observer.copyright;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class TextModelCorrect2 implements TextModel {
	
	@FunctionalInterface
	private static interface TextCommand {
		abstract void execute();
	}

	private final List<TextCommand> queue = new LinkedList<>();

	private int notifyLevel = 0;

	private void handle(TextCommand c) {		
		notifyLevel++;
		if(notifyLevel > 1) {
			queue.add(c);
		}
		else {
			c.execute();
			while(!queue.isEmpty()) {
				queue.remove(0).execute();
			}
		}
		notifyLevel--;
	}

	
	private final StringBuilder text = new StringBuilder();
	private final List<Listener> listeners = new ArrayList<>();

	@Override
	public void addListener(Listener l) {
		listeners.add(l);
	}

	@Override
	public void insert(final int pos, final char ch) {
		handle(() -> {
			if (pos < 0 || pos > text.length())
				throw new IllegalArgumentException();
			text.insert(pos, ch);
			for (Listener l : listeners) {
				l.notifyInsert(pos, ch);
			}
		});
	}

	@Override
	public void delete(final int from, final int len) {
		handle(() -> {
			if (from < 0 || len < 0 || from + len > text.length())
				throw new IllegalArgumentException();
			text.delete(from, from + len);
			for (Listener l : listeners) {
				l.notifyDelete(from, len);
			}
		});
	}

	@Override
	public String getSubstring(int from, int len) {
		if (from < 0 || len < 0 || from + len > text.length()) throw new IllegalArgumentException();
		return text.substring(from, from + len);
	}

	@Override
	public String toString() {
		return text.toString();
	}
}
