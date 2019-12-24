package patterns.command.colorframe;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.KeyStroke;
import javax.swing.WindowConstants;

@SuppressWarnings("serial")
public class ColorCommandFrameAction extends JFrame {

	public static void main(String[] args) {
		JFrame f = new ColorCommandFrameAction();
		f.setSize(400, 300);
		f.setVisible(true);
	}

	private final JPanel coloredPanel = new JPanel();

	class YellowAction extends AbstractAction {
		{
			this.putValue(Action.NAME, "Yellow");
			this.putValue(Action.SHORT_DESCRIPTION, "Set background to yellow");
			this.putValue(Action.SMALL_ICON, new ImageIcon(this.getClass().getResource("/yellow_bullet.gif")));
			this.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control Y"));
			this.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_Y);
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			coloredPanel.setBackground(Color.YELLOW);
			yellowAction.setEnabled(false);
			redAction.setEnabled(true);
		}
	}

	class RedAction extends AbstractAction {
		{
			this.putValue(Action.NAME, "Red");
			this.putValue(Action.SHORT_DESCRIPTION, "Set background to red");
			this.putValue(Action.SMALL_ICON, new ImageIcon(this.getClass().getResource("/red_bullet.gif")));
			this.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control R"));
			this.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_R);
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			coloredPanel.setBackground(Color.RED);
			yellowAction.setEnabled(true);
			redAction.setEnabled(false);
		}
	}

	private final YellowAction yellowAction = new YellowAction();
	private final RedAction redAction = new RedAction();

	public ColorCommandFrameAction() {
		super("ColorCommandFrameAction");
		setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		initButtons();
		initPopupMenu();
		initMenu();
	}

	public void initButtons() {
		coloredPanel.add(new JButton(yellowAction));
		coloredPanel.add(new JButton(redAction));
		add(coloredPanel);
	}

	public void initPopupMenu() {
		final JPopupMenu popupMenu = new JPopupMenu();
		popupMenu.add(yellowAction);
		popupMenu.add(redAction);

		coloredPanel.addMouseListener(new MouseAdapter() {
			public @Override void mousePressed(MouseEvent e) {
				maybeShowPopup(e);
			}

			public @Override void mouseReleased(MouseEvent e) {
				maybeShowPopup(e);
			}

			private void maybeShowPopup(MouseEvent e) {
				if (e.isPopupTrigger()) {
					popupMenu.show(e.getComponent(), e.getX(), e.getY());
				}
			}
		});
	}

	public void initMenu() {
		JMenuBar menuBar = new JMenuBar();
		JMenu colorMenu = new JMenu("Color");
		colorMenu.setMnemonic(KeyEvent.VK_C);

		colorMenu.add(yellowAction);
		colorMenu.add(redAction);
		menuBar.add(colorMenu);
		setJMenuBar(menuBar);
	}

}
