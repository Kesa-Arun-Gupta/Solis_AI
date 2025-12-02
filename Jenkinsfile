import groovy.json.*
pipeline {
    agent { label "platform" }

    environment {
        IMAGE_NAME = "solis_ai"
        ARTIFACTORY_SERVER_ID = "artifactory-east"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

       stage('Docker Build') {
            steps {
                script {
                    def tag = "${BUILD_NUMBER}"
                    def image = "artifactory-east.corp.lumsb.com/lumsb/solis_ai:${tag}"

                    echo "Building Docker image: ${image}"
                    sh "docker build -t ${image} ."
                }
            }
        }
        stage('Docker Push') {
            steps {
                script {
                    def tag = "${BUILD_NUMBER}"
                    def image = "artifactory-east.corp.lumsb.com/lumsb/solis_ai:${tag}"

                    echo "Pushing Docker image: ${image}"
                    sh "docker push ${image}"
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
