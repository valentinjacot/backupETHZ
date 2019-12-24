package patterns.factory.gui;

import java.awt.GridLayout;
import java.awt.LayoutManager;
import java.awt.event.ActionEvent;

import patterns.factory.gui.Components.ActionListener;
import patterns.factory.gui.Components.Component;

@SuppressWarnings("serial")
public class ComponentsSwing {

	public static class ButtonSwing extends javax.swing.JButton implements Components.Button {
		ButtonSwing(String label, Components.ActionListener listener) {
			super(label);
			addActionListener(listener);
		}

		public void addActionListener(final ActionListener listener) {
			this.addActionListener(new java.awt.event.ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {
					listener.actionPerformed(ButtonSwing.this);
				}
			});
		}
	}

	public static class LabelSwing extends javax.swing.JLabel implements Components.Label {
		LabelSwing(String label) {
			super(label);
		}
	}

	public static class FieldSwing extends javax.swing.JTextField implements Components.Field {
		public FieldSwing(int width, boolean enabled) {
			super(width);
			setEnabled(enabled);
		}
	}

	public static class FrameSwing extends javax.swing.JFrame implements Components.Frame {
		FrameSwing(String title) {
			super(title);
			setResizable(false);
			setDefaultCloseOperation(EXIT_ON_CLOSE);
		}

		@Override
		public void add(Component c) {
			add((java.awt.Component) c);
		}

		@Override
		public void setGrid(int w, int h) {
			LayoutManager m = new GridLayout(w, h);
			if (isRootPaneCheckingEnabled()) {
				getContentPane().setLayout(m);
			} else {
				super.setLayout(m);
			}
		}

		public @Override void setVisible(boolean visible) {
			super.pack();
			super.setVisible(visible);
		}
	}
}
