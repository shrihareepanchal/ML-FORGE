# ML-FORGE Deployment Guide

This guide provides step-by-step instructions for deploying ML-FORGE to various platforms.

## GitHub Setup

1. **Create a GitHub Repository**:
   - Go to [GitHub](https://github.com) and sign in
   - Click the "+" icon in the top right and select "New repository"
   - Name your repository (e.g., "ml-xpert")
   - Add a description
   - Choose public or private visibility
   - Click "Create repository"

2. **Initialize Git and Push Your Code**:
   ```bash
   # Initialize Git in your project directory
   git init
   
   # Add all files to staging
   git add .
   
   # Commit the changes
   git commit -m "Initial commit"
   
   # Add the remote repository
   git remote add origin https://github.com/yourusername/ml-xpert.git
   
   # Push to GitHub
   git push -u origin main
   ```

## Deployment Options

### 1. Streamlit Cloud (Recommended for Simplicity)

1. **Sign up for Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account

2. **Deploy Your App**:
   - Click "New app"
   - Select your repository, branch, and main file path (app.py)
   - Click "Deploy"
   - Your app will be available at a URL like: https://yourusername-ml-xpert-app-xyz.streamlit.app

3. **Advanced Configuration** (if needed):
   - You can set secrets and environment variables in the Streamlit Cloud dashboard
   - For more details, see [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-cloud)

### 2. Heroku Deployment

1. **Sign up for Heroku**:
   - Create an account at [Heroku](https://signup.heroku.com/)
   - Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)

2. **Login to Heroku and Create an App**:
   ```bash
   # Login to Heroku
   heroku login
   
   # Create a new Heroku app
   heroku create ml-xpert
   ```

3. **Deploy to Heroku**:
   ```bash
   # Push to Heroku
   git push heroku main
   
   # Scale the app
   heroku ps:scale web=1
   ```

4. **Open Your App**:
   ```bash
   heroku open
   ```

5. **Troubleshooting**:
   - Check logs with `heroku logs --tail`
   - Ensure your `requirements.txt`, `Procfile`, and `setup.sh` are correctly configured

### 3. AWS Elastic Beanstalk

1. **Install AWS CLI and EB CLI**:
   ```bash
   pip install awscli awsebcli
   ```

2. **Configure AWS**:
   ```bash
   aws configure
   ```

3. **Initialize EB Application**:
   ```bash
   eb init -p python-3.10 ml-xpert
   ```

4. **Create an Environment and Deploy**:
   ```bash
   eb create ml-xpert-env
   ```

5. **Open Your Application**:
   ```bash
   eb open
   ```

### 4. Google Cloud Run

1. **Install Google Cloud SDK**:
   - Follow instructions at [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

2. **Initialize Google Cloud**:
   ```bash
   gcloud init
   ```

3. **Build a Docker Container**:
   - Create a `Dockerfile` in your project root:
     ```Dockerfile
     FROM python:3.10-slim
     
     WORKDIR /app
     
     COPY . .
     
     RUN pip install -r requirements.txt
     
     EXPOSE 8080
     
     CMD streamlit run --server.port 8080 --server.enableCORS false app.py
     ```

4. **Build and Deploy**:
   ```bash
   # Build the container
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ml-xpert
   
   # Deploy to Cloud Run
   gcloud run deploy ml-xpert --image gcr.io/YOUR_PROJECT_ID/ml-xpert --platform managed
   ```

## Important Considerations

### Data Storage
- The app currently uses local file storage which is not persistent in cloud deployments
- For production, consider using cloud storage solutions:
  - AWS S3
  - Google Cloud Storage
  - Azure Blob Storage

### Security
- Ensure sensitive data is not committed to GitHub
- Use environment variables for API keys and credentials
- Consider adding authentication for your deployed app

### Performance
- Large ML models may require more memory than free tiers provide
- Consider optimizing your models or upgrading your deployment plan

### Costs
- Monitor your usage to avoid unexpected charges
- Set up billing alerts
- Start with free tiers to test before scaling up

## Maintenance

1. **Updates and Improvements**:
   ```bash
   # Pull latest changes from GitHub
   git pull origin main
   
   # Make your changes
   # ...
   
   # Commit and push
   git add .
   git commit -m "Update description"
   git push origin main
   ```

2. **Redeploy**:
   - For Streamlit Cloud: Automatic redeployment on push to GitHub
   - For Heroku: `git push heroku main`
   - For AWS EB: `eb deploy`
   - For Google Cloud Run: Rebuild and redeploy container

## Need Help?

If you encounter issues during deployment, check:
- Platform-specific documentation
- Stack Overflow for common errors
- GitHub Issues for this project
- Contact the maintainer at your.email@example.com 