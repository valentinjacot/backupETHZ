package patterns.command.matmul;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * This class multiplies two N by N matrices.
 */
public final class MatMult {
	/** Dimension of the matrices. */
	private static final int N = 1000;
	
	/** MatMult is a utility class, no instances are ever created. */
	private MatMult() { }

	/** computes one row of the product a*b. */
	static class ComputeRow implements Runnable {
		private final int row;
		private final int[][] a, b, c;

		public ComputeRow(int row, int[][] a, int[][] b, int[][] c) { 
			this.row = row; 
			this.a = a; this.b = b; this.c = c;
		}

		@Override
		public void run() {
			for (int i = 0; i < N; i++) {
				int sum = 0;
				for (int j = 0; j < N; j++) {
					sum += a[row][j] * b[j][i];					
				}
				c[row][i] = sum;
			}
		}
	}
	
	public static void main(String[] args) throws Exception {
		int[][] a = new int[N][N];
		int[][] b = new int[N][N];
		int[][] c = new int[N][N];
		
		createRandomMatrix(a);
		createRandomMatrix(b);
		
		System.out.println("starting");
		System.out.println(Runtime.getRuntime().availableProcessors());
		
		long start = System.currentTimeMillis();
		computeProductSequential(a, b, c);
		long end = System.currentTimeMillis();
		System.out.println("seq done [" + (end-start) + " msec]");
		
		int nOfThreads = 1;
		while(nOfThreads <= Runtime.getRuntime().availableProcessors()) {
			start = System.currentTimeMillis();
			computeProductParallel(nOfThreads, a, b, c);
			end = System.currentTimeMillis();
			System.out.printf("par[%d] done [%5d msec]\n", nOfThreads, (end-start));
			nOfThreads = 2*nOfThreads;
		}
	}

	/**
	 * Initializes the given matrix with random values. The matrix may be jagged.
	 * @param mat matrix to be defined.
	 */
	private static void createRandomMatrix(final int[][] mat) {
		Random r = new Random();
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[i].length; j++) {
				mat[i][j] = r.nextInt();
			}
		}
	}

	private static void computeProductSequential(final int[][] a, final int[][] b, final int[][] c) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				int sum = 0;
				for (int k = 0; k < N; k++) {
					sum += a[i][k] * b[k][j];
				}
				c[i][j] = sum;
			}
		}
	}

	private static void computeProductParallel(int nOfThreads, final int[][] a,
			final int[][] b, final int[][] c) throws InterruptedException {
		ExecutorService ex = Executors.newFixedThreadPool(nOfThreads);
		//ExecutorService ex = Executors.newCachedThreadPool();
		for (int i = 0; i < N; i++) {
			ex.execute(new ComputeRow(i, a, b, c));
		}
		ex.shutdown();
		ex.awaitTermination(1, TimeUnit.HOURS);
	}
	
}

