package patterns.observer.copyright;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

public class TextModelCorrect1 implements TextModel {
	
	@FunctionalInterface
	private static interface TextCommand {
		void execute();
	}
	
	private final Deque<TextCommand> queue = new LinkedList<>();

	private boolean isCommandRunning = false;

	private void processQueue() {
		if (!isCommandRunning) {
			TextCommand cmd = queue.peekLast();
			if (cmd != null) {
				queue.pollLast();
				cmd.execute();
			}
		}
	}

	private final StringBuilder text = new StringBuilder();
	private final List<Listener> listeners = new ArrayList<>();

	@Override
	public void addListener(Listener l) {
		listeners.add(l);
	}

	@Override
	public void insert(final int pos, final char ch) {
		queue.addFirst(() -> {
			if (pos < 0 || pos > text.length())
				throw new IllegalArgumentException();
			text.insert(pos, ch);
			isCommandRunning = true;
			for (Listener l : listeners) {
				l.notifyInsert(pos, ch);
			}
			isCommandRunning = false;
			processQueue(); // invokes the next command in the queue
		});
		processQueue();
	}

	@Override
	public void delete(final int from, final int len) {
		queue.addFirst(() -> {
			if (from < 0 || len < 0 || from + len > text.length())
				throw new IllegalArgumentException();
			text.delete(from, from + len);
			isCommandRunning = true;
			for (Listener l : listeners) {
				l.notifyDelete(from, len);
			}
			isCommandRunning = false;
			processQueue();
		});
		processQueue();
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
