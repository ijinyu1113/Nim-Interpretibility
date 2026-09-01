from huggingface_hub import HfApi
api = HfApi()
repo = "ijinyu1113/test-delete-me"
api.create_repo(repo, exist_ok=True, repo_type="model")
api.create_branch(repo, branch="test-branch")
print("SUCCESS: repo created + branch created")
api.delete_repo(repo)
print("Cleaned up test repo")
