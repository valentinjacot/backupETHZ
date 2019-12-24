package jdraw.constrainers;

import java.awt.Point;

import jdraw.framework.DrawContext;
import jdraw.framework.DrawGrid;

/**
 * @author Max Rossmannek
 */

public abstract class AbstractGrid implements DrawGrid {

    protected final DrawContext context;
    private final int stepX;
    private final int stepY;
    private final String name;

    public AbstractGrid(DrawContext c, String n, int dx, int dy) {
        this.context = c;
        this.name = n;
        this.stepX = dx;
        this.stepY = dy;
    }

    public String getName() {
        return name;
    }

    public Point constrainPoint(Point p) {
        int newX = stepX * (Math.round(p.x / stepX));
        int newY = stepY * (Math.round(p.y / stepY));
        return new Point(newX, newY);
    }

    public int getStepX(boolean right) {
        return stepX;
    }

    public int getStepY(boolean down) {
        return stepY;
    }

    public void activate() {
        this.context.showStatusText("Activated " + name);
    }

    public void deactivate() {
        this.context.showStatusText("Deactivated " + name);
    }

    public void mouseDown() { }

    public void mouseUp() { }
}
