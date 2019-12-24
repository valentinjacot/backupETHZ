package patterns.factory.gui;

import org.eclipse.swt.SWT;
import org.eclipse.swt.events.SelectionEvent;
import org.eclipse.swt.events.SelectionListener;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;
import org.eclipse.swt.widgets.Text;

import patterns.factory.gui.Components.ActionListener;
import patterns.factory.gui.Components.Button;
import patterns.factory.gui.Components.Component;
import patterns.factory.gui.Components.Field;
import patterns.factory.gui.Components.Frame;
import patterns.factory.gui.Components.Label;

public class ComponentsSWT {
	private static interface ComponentSWT {
		void createPeer(Shell shell);
	}

	public static class ButtonSWT implements Button, ComponentSWT {
		String label;
		ActionListener listener;

		ButtonSWT(String label, final ActionListener listener) {
			this.label = label;
			this.listener = listener;
		}

		@Override
		public void createPeer(Shell shell) {
			org.eclipse.swt.widgets.Button b = new org.eclipse.swt.widgets.Button(
					shell, SWT.PUSH);
			b.setText(label);
			b.addSelectionListener(new SelectionListener() {
				@Override
				public void widgetSelected(SelectionEvent e) {
					listener.actionPerformed(ButtonSWT.this);
				}

				@Override
				public void widgetDefaultSelected(SelectionEvent e) {}
			});
		}
	}

	public static class LabelSWT implements Label, ComponentSWT {
		private String label;

		LabelSWT(String label) {
			this.label = label;
		}

		@Override
		public void createPeer(Shell shell) {
			org.eclipse.swt.widgets.Label l = new org.eclipse.swt.widgets.Label(
					shell, SWT.NULL);
			l.setText(label);
		}
	}

	public static class FieldSWT implements Field, ComponentSWT {
		private boolean enabled;
		private Text t;

		FieldSWT(boolean enabled) {
			this.enabled = enabled;
		}

		@Override
		public void createPeer(Shell shell) {
			t = new Text(shell, SWT.SINGLE);
			t.setEnabled(enabled);
		}

		@Override
		public String getText() {
			return t.getText();
		}

		@Override
		public void setText(String text) {
			t.setText(text);
		}
	}

	public static class FrameSWT implements Frame {
		final Display display = new Display();
		final Shell shell = new Shell(display, SWT.SHELL_TRIM & (~SWT.RESIZE));

		FrameSWT(String title) {
			shell.setText(title);
		}

		@Override
		public void setVisible(boolean visible) {
			shell.setSize(180, 135);
			shell.open();
			while (!shell.isDisposed()) {
				if (!display.readAndDispatch()) display.sleep();
			}
		}

		@Override
		public void add(Component c) {
			if (c instanceof ComponentSWT) {
				((ComponentSWT)c).createPeer(shell);
			}
		}

		@Override
		public void setGrid(int w, int h) {
			GridLayout g = new GridLayout();
			g.numColumns = h;
			shell.setLayout(g);
		}
	}
}
