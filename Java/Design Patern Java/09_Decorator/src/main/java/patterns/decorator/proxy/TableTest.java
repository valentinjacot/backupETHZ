package patterns.decorator.proxy;

import java.lang.reflect.Proxy;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableModel;

@SuppressWarnings("serial")
public class TableTest extends JFrame {

	public static void main(String[] args) {
		JFrame frame = new TableTest();
		frame.setTitle("Tables and Models");
		frame.setBounds(300,300,450,300);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
	}

	public TableTest() {
		TableModel model = new TestModel();
		model = (TableModel)Proxy.newProxyInstance(
				TableModel.class.getClassLoader(), 
				new Class[] { TableModel.class },
				new LoggingHandler(model)
		);

		add(new JScrollPane(new JTable(model)));
	}

	private static class TestModel extends AbstractTableModel {
		final int rows = 100, cols = 10;
		@Override public int getRowCount() { return rows; }
		@Override public int getColumnCount() { return cols; }
		@Override public Object getValueAt(int row, int col) {
			return "(" + row + "," + col + ")";
		}
	}
}

