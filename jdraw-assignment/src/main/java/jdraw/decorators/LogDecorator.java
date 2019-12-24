package jdraw.decorators;

import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;

import java.util.List;

import jdraw.framework.Figure;
import jdraw.framework.FigureEvent;
import jdraw.framework.FigureHandle;
import jdraw.framework.FigureListener;

/**
 * @author Max Rossmannek
 */
@SuppressWarnings("serial")
public class LogDecorator extends AbstractDecorator {

    public LogDecorator(Figure f) {
        super(f);
    }

    @Override
    public void draw(Graphics g) {
        System.out.println("Calling draw() on Figure " + inner);
        super.draw(g);
    }

    @Override
    public void move(int dx, int dy) {
        System.out.println("Calling move() on Figure " + inner);
        super.move(dx, dy);
    }

    @Override
    public boolean contains(int x, int y) {
        System.out.println("Calling contains() on Figure " + inner);
        return super.contains(x, y);
    }

    @Override
    public void setBounds(Point origin, Point corner) {
        System.out.println("Calling setBounds() on Figure " + inner);
        super.setBounds(origin, corner);
    }

    @Override
    public Rectangle getBounds() {
        System.out.println("Calling getBounds() on Figure " + inner);
        return super.getBounds();
    }

    @Override
    public List<FigureHandle> getHandles() {
        System.out.println("Calling getHandles() on Figure " + inner);
        return super.getHandles();
    }


    @Override
    public void addFigureListener(FigureListener listener) {
        super.addFigureListener(listener);
        System.out.println("Calling addFigureListener() on Figure " + inner);
    }

    @Override
    public void removeFigureListener(FigureListener listener) {
        super.removeFigureListener(listener);
        System.out.println("Calling removeFigureListener() on Figure " + inner);
    }

    @Override
    public LogDecorator clone() {
        System.out.println("Calling clone() on Figure " + inner);
        LogDecorator copy = (LogDecorator) super.clone();
        return copy;
    }
}
