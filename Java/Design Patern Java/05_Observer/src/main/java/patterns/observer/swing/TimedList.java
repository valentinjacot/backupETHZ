package patterns.observer.swing;

import javax.swing.AbstractListModel;
import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JScrollPane;
import javax.swing.ListSelectionModel;
import javax.swing.SwingUtilities;

@SuppressWarnings("serial")
public class TimedList extends JFrame {

	private TimedList() {
		class Model extends AbstractListModel<String> {
			private int size = 5;

			@Override
			public int getSize() {
				return size;
			}

			@Override
			public String getElementAt(int n) {
				return "" + n;
			}

			public void incrementSize() {
				size++;
				fireIntervalAdded(this, size, size);
			}
		}
		final Model model = new Model();

		Thread t = new Thread(() -> {
			while (true) {
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				model.incrementSize();
			}
		});
		t.start();

		JList<String> list = new JList<>(model);
		list.setVisibleRowCount(20);
		list.setFixedCellHeight(18);
		list.setFixedCellWidth(500);
		list.setSelectionMode(ListSelectionModel.SINGLE_INTERVAL_SELECTION);

		add(new JScrollPane(list));

		setDefaultCloseOperation(EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		SwingUtilities.invokeLater(() -> {
			JFrame f = new TimedList();
			f.pack();
			f.setVisible(true);
		});
	}
}
