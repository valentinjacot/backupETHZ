package patterns.observer.copyright;

import java.util.ArrayList;
import java.util.List;

import javax.swing.SwingUtilities;

/*
 * This solution is motivated by the stackoverflow entry
 * http://stackoverflow.com/questions/3599519/in-java-swing-is-there-a-way-to-legally-attempt-to-mutate-in-notification
 */
public class TextModelCorrect3 implements TextModel {
	private final StringBuilder text = new StringBuilder();
	private final List<Listener> listeners = new ArrayList<>();

	@Override
	public void addListener(Listener l) {
		listeners.add(l);
	}

	@Override
	public void insert(final int pos, final char ch) {
		if (pos < 0 || pos > text.length())
			throw new IllegalArgumentException();
		text.insert(pos, ch);
		for (Listener l : listeners) {
			SwingUtilities.invokeLater(() -> l.notifyInsert(pos, ch));
		}
	}

	@Override
	public void delete(int from, int len) {
		if (from < 0 || len < 0 || from + len > text.length())
			throw new IllegalArgumentException();
		text.delete(from, from + len);
		for (Listener l : listeners) {
			SwingUtilities.invokeLater(() -> l.notifyDelete(from, len));
		}
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
