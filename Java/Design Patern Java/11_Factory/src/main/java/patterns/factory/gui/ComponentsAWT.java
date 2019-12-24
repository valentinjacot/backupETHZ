package patterns.factory.gui;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import patterns.factory.gui.Components.ActionListener;
import patterns.factory.gui.Components.Component;

@SuppressWarnings("serial")
public class ComponentsAWT {

	public static class ButtonAWT extends java.awt.Button implements Components.Button {
		ButtonAWT(String label, Components.ActionListener listener) {
			super(label);
			addActionListener(listener);
		}

		public void addActionListener(final ActionListener listener) {
			this.addActionListener(new java.awt.event.ActionListener() {
				@Override
				public void actionPerformed(ActionEvent e) {
					listener.actionPerformed(ButtonAWT.this);
				}
			});
		}
	}

	static class FieldAWT extends java.awt.TextField implements Components.Field {
		FieldAWT(int width, boolean enabled) {
			super(width);
			setEnabled(enabled);
		}
	}

	static class LabelAWT extends java.awt.Label implements Components.Label {
		LabelAWT(String text) {
			super(text);
		}
	}

	static class FrameAWT extends java.awt.Frame implements Components.Frame {
		FrameAWT(String label) {
			super(label);
			setResizable(false);
			addWindowListener(new WindowAdapter() {
				@Override
				public void windowClosing(WindowEvent e) {
					System.exit(0);
				}
			});
		}

		@Override
		public void add(Component c) {
			add((java.awt.Component) c);
		}

		@Override
		public void setVisible(boolean visible) {
			pack();
			super.setVisible(visible);
		}

		@Override
		public void setGrid(int w, int h) {
			setLayout(new GridLayout(4, 2));
		}
	}
}
