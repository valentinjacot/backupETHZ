package classesAbstraites;

abstract class Animal {
  
        /**
         * La couleur de l'animal
         */
        protected String couleur;
        /**
         * Le poids
         */
        protected int poids;
         
        /**
         * La m�thode manger
         */
        protected void manger(){
                System.out.println("Je mange de la viande");
        }
         
        /**
         * La m�thode boire
         */
        protected void boire(){
                System.out.println("Je bois de l'eau !");
        }
         
        /**
         * La m�thode de d�placement
         */
        abstract void deplacement();
        /**
         * La m�thode de cri
         */
        abstract void crier();
         
        public String toString(){
                 
                String str = "Je suis un objet de la " + this.getClass() + ", je suis " + this.couleur + ", je p�se " + this.poids;
                return str;
        }
         
}