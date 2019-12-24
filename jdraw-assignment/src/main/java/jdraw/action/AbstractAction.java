package jdraw.action;

import java.awt.event.ActionEvent;

import javax.swing.event.MenuEvent;
import javax.swing.event.MenuListener;

import jdraw.framework.DrawContext;

@SuppressWarnings("serial")
public abstract class AbstractAction extends javax.swing.AbstractAction implements MenuListener {
	protected DrawContext actionContext;
    protected static final String IMAGES = "/images/";

	public AbstractAction(DrawContext AC) {
		actionContext= AC;
	}
	@Override
	public abstract void actionPerformed(ActionEvent e);

	@Override
	public void menuSelected(MenuEvent e) {
		setEnabled(true);
	}

	@Override
	public void menuDeselected(MenuEvent e) {
		setEnabled(true);
	}

	@Override
	public void menuCanceled(MenuEvent e) {
		setEnabled(true);
	}

}
